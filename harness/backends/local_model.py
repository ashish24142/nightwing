"""
local_model.py — local fine-tuned backend (P2.4): windowed inference.

Same Backend interface and same prompt.py framing as the frontier backends, so
the pilot is scored on the IDENTICAL harness (rule #3). Because the pilot is
TRAINED on sliding windows (data/prepare_train.py), it is also EVALUATED on the
same windows here: each contract is split with harness/windowing.py, every
(window, question) is asked, and predicted spans are aggregated across windows.

Inference uses transformers + peft (portable: works on the Windows/Blackwell
smoke box and the Linux cloud box). Heavy deps import lazily in __init__, so
importing this module never requires a GPU stack.

Config (config/models.yaml backends.local):
    base_model, adapter_path, max_new_tokens, load_in_4bit, win_chars, overlap_chars
"""
from __future__ import annotations

from .. import prompt as P
from ..windowing import iter_windows
from .base import Backend, ContractResult, Question, Usage


class LocalBackend(Backend):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_model = cfg["base_model"]
        self.adapter_path = cfg.get("adapter_path", "")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 256))
        self.win_chars = int(cfg.get("win_chars", 8000))
        self.overlap_chars = int(cfg.get("overlap_chars", 2000))
        self.gen_batch_size = int(cfg.get("gen_batch_size", 8))
        self.model_id = f"local:{self.base_model.split('/')[-1]}"
        if self.adapter_path:
            self.model_id += f"+{str(self.adapter_path).rstrip('/').split('/')[-1]}"

        # -- lazy GPU imports --
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant = None
        if cfg.get("load_in_4bit", False):
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                quant = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True)
            except ImportError:
                print("   local: load_in_4bit set but bitsandbytes missing -> bf16")

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for batched decoder generation
        # token ids for the JSON boolean after '"present": ' — used to read a
        # CALIBRATED P(present) from the logits (vs the model's self-reported
        # confidence, which produces a degenerate PR curve / P@R=0)
        def _first_ids(variants):
            ids = set()
            for v in variants:
                enc = self.tokenizer.encode(v, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            return ids
        self._true_ids = _first_ids(["true", " true", "True", " True"])
        self._false_ids = _first_ids(["false", " false", "False", " False"])
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model, quantization_config=quant,
                torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation="sdpa")
        except (ValueError, ImportError):
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model, quantization_config=quant,
                torch_dtype=torch.bfloat16, device_map="auto")
        if self.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self.model = model
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------
    def _generate_batch(self, users: list[str]) -> list[tuple[str, float | None]]:
        """Greedy-generate for a batch of user prompts (shared SYSTEM_PROMPT).
        Left-padded so all inputs share one length and gen tokens align.
        Returns (text, p_present) per prompt: p_present is the softmax mass on
        'true' vs 'false' at the JSON boolean position — a real, graded
        confidence for the PR sweep (None if no boolean token was emitted)."""
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": P.SYSTEM_PROMPT},
                 {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True)
            for u in users
        ]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.cfg.get("max_seq_length", 4096))
        enc = {k: v.to(self.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        with self.torch.no_grad():
            # ponytail: output_scores keeps [batch, vocab] per gen step on GPU
            # (~0.5GB at batch 8); fine on A100, lower gen_batch_size if tight
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[:, in_len:]
        texts_out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        bool_ids = self._true_ids | self._false_ids
        results = []
        for i, text in enumerate(texts_out):
            p_present = None
            for step, tok in enumerate(gen[i].tolist()):
                if tok in bool_ids:
                    probs = self.torch.softmax(out.scores[step][i].float(), dim=-1)
                    p_t = sum(probs[t].item() for t in self._true_ids)
                    p_f = sum(probs[t].item() for t in self._false_ids)
                    if p_t + p_f > 0:
                        p_present = p_t / (p_t + p_f)
                    break
            results.append((text, p_present))
        return results

    def _run_jobs(self, users: list[str]) -> list[tuple[str, float | None] | None]:
        """Batch through all prompts; on OOM, halve the batch / fall back to 1."""
        outs: list[tuple[str, float | None] | None] = []
        bs = self.gen_batch_size
        i = 0
        while i < len(users):
            chunk = users[i:i + bs]
            try:
                outs.extend(self._generate_batch(chunk))
                i += len(chunk)
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if bs == 1:  # can't shrink further -> mark this one failed, move on
                    outs.append(None)
                    i += 1
                else:
                    bs = max(1, bs // 2)  # retry the same slice with a smaller batch
        return outs

    def predict_contract(self, contract_text: str,
                         questions: list[Question]) -> ContractResult:
        windows = iter_windows(contract_text, self.win_chars, self.overlap_chars)
        res = ContractResult()
        if not windows:
            return res
        # build every (question, window) prompt, then batch-generate
        jobs = []  # (qid, user_prompt)
        for q in questions:
            qtext = P.build_question_text(q.question, q.category)
            for w in windows:
                jobs.append((q.qid, P.build_contract_block(w.text) + "\n\n" + qtext))
        try:
            outs = self._run_jobs([u for _, u in jobs])
        except Exception as e:  # whole-contract generation failure
            print(f"      ! local contract gen failed: {type(e).__name__}: {str(e)[:100]}")
            res.errors = len(questions)
            for q in questions:
                res.nbest[q.qid] = []
            return res

        # aggregate spans across windows per qid; confidence = logprob-based
        # P(present) when available (graded -> real PR curve), else self-reported
        agg = {q.qid: {"present": False, "spans": [], "conf": 0.0} for q in questions}
        failed_qids = set()
        for (qid, _), out in zip(jobs, outs):
            if out is None:
                failed_qids.add(qid)
                continue
            text, p_present = out
            parsed = P.parse_response(text)
            if parsed["present"] and parsed["spans"]:
                a = agg[qid]
                a["present"] = True
                for s in parsed["spans"]:
                    if s not in a["spans"]:
                        a["spans"].append(s)
                conf = p_present if p_present is not None else parsed["confidence"]
                a["conf"] = max(a["conf"], conf)
        for q in questions:
            a = agg[q.qid]
            res.nbest[q.qid] = P.to_nbest(
                {"present": a["present"], "spans": a["spans"], "confidence": a["conf"]})
        res.errors = len(failed_qids)  # qids where every window OOM'd at batch=1
        res.usage = Usage()  # local inference is free; no $ tracking
        return res
