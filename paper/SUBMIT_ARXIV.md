# arXiv submission guide (you do these steps; they need your account)

## What to upload

Two files from this directory (arXiv compiles the LaTeX itself):

- `main.tex`
- `scaling_curve.pdf`

The locally compiled `main.pdf` (6 pages) is for your review only; do not
upload it (arXiv wants source).

## Steps

1. Account: https://arxiv.org/user/ (register with your email; use your real
   name, it becomes the immutable author record).
2. Endorsement: first-time submitters to a category may need an endorsement
   for **cs.CL** (Computation and Language). If prompted, the system gives
   you a code to send to an endorser; institutional email addresses often
   skip this entirely. This can take a day or two, so start the account now.
3. Start a new submission: https://arxiv.org/submit
   - License: recommended **CC BY 4.0** (matches the repo's openness).
   - Primary category: **cs.CL**. Optional cross-list: cs.LG.
   - Upload `main.tex` and `scaling_curve.pdf` together (zip or individually).
   - arXiv compiles; preview the PDF and confirm it matches `main.pdf`.
4. Metadata:
   - Title: copy from the paper.
   - Abstract: copy from the paper (plain text, no LaTeX macros needed
     except $\sigma$ style math is fine).
   - Authors: Ashish Kumar Singh.
5. Submit. Moderation typically takes 1-3 business days; you get an arXiv id
   (2507.xxxxx) and a permanent URL.

## After it is live

- Add the arXiv link to: repo README header line, both essays' bylines, the
  HuggingFace model card, and your LinkedIn.
- The v3 (recipe-complete) run extends this paper into the NLLP @ EMNLP 2026
  submission (deadline Aug 11, via OpenReview / ARR commitment). NLLP uses
  the ACL template, so the port is mechanical; the content arc becomes:
  pre-registered miss (this paper) -> controlled recipe fix -> resolution.
