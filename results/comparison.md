# nightwing — pilot vs frontier (official CUAD AUPR, full test split)

## Overall

| Model | AUPR | P@80R | P@90R | gap to best frontier |
|---|---|---|---|---|
| claude-opus-4-8 | **0.561** | 0.000 | 0.000 | +0.000 |
| claude-opus-4-6 | **0.498** | 0.000 | 0.000 | -0.063 |
| gpt-5.2 | **0.423** | 0.000 | 0.000 | -0.138 |
| gpt-4o | **0.421** | 0.000 | 0.000 | -0.140 |
| local:Qwen3-14B+checkpoint-1250 *(pilot)* | **0.291** | 0.000 | 0.000 | -0.270 |

## Per-category (40 scorable; 'Price Restrictions' excluded — 0 positives in test)

| Category | local:Qwen3-14B+checkpoint-1250 | best frontier | (model) | Δ |
|---|---|---|---|---|
| Agreement Date 🏆 | 0.687 | 0.168 | claude-opus-4-6 | +0.520 |
| Effective Date 🏆 | 0.369 | 0.154 | gpt-4o | +0.214 |
| Volume Restriction | 0.008 | 0.038 | gpt-5.2 | -0.031 |
| Post-Termination Services | 0.053 | 0.103 | claude-opus-4-8 | -0.050 |
| Non-Transferable License | 0.442 | 0.503 | claude-opus-4-8 | -0.061 |
| Irrevocable Or Perpetual License | 0.594 | 0.659 | claude-opus-4-8 | -0.065 |
| Expiration Date | 0.633 | 0.708 | claude-opus-4-8 | -0.076 |
| Minimum Commitment | 0.162 | 0.257 | claude-opus-4-6 | -0.094 |
| Third Party Beneficiary | 0.585 | 0.695 | claude-opus-4-8 | -0.110 |
| Insurance | 0.226 | 0.353 | claude-opus-4-6 | -0.127 |
| Termination For Convenience | 0.463 | 0.618 | claude-opus-4-8 | -0.155 |
| Affiliate License-Licensee | 0.398 | 0.557 | claude-opus-4-6 | -0.159 |
| License Grant | 0.365 | 0.528 | claude-opus-4-8 | -0.164 |
| Non-Disparagement | 0.229 | 0.396 | gpt-4o | -0.167 |
| Document Name | 0.711 | 0.887 | claude-opus-4-8 | -0.176 |
| Affiliate License-Licensor | 0.210 | 0.405 | claude-opus-4-8 | -0.196 |
| Rofr/Rofo/Rofn | 0.213 | 0.411 | claude-opus-4-8 | -0.198 |
| Change Of Control | 0.164 | 0.371 | gpt-4o | -0.207 |
| Ip Ownership Assignment | 0.143 | 0.351 | gpt-5.2 | -0.208 |
| Revenue/Profit Sharing | 0.181 | 0.423 | claude-opus-4-8 | -0.242 |
| Exclusivity | 0.215 | 0.465 | claude-opus-4-6 | -0.250 |
| Governing Law | 0.669 | 0.924 | claude-opus-4-8 | -0.255 |
| No-Solicit Of Customers | 0.296 | 0.573 | claude-opus-4-6 | -0.278 |
| Audit Rights | 0.147 | 0.441 | claude-opus-4-8 | -0.294 |
| Cap On Liability | 0.194 | 0.523 | claude-opus-4-8 | -0.329 |
| Joint Ip Ownership | 0.184 | 0.524 | claude-opus-4-6 | -0.340 |
| Warranty Duration | 0.053 | 0.404 | claude-opus-4-8 | -0.351 |
| Non-Compete | 0.164 | 0.517 | claude-opus-4-8 | -0.353 |
| Anti-Assignment | 0.224 | 0.581 | claude-opus-4-8 | -0.358 |
| Renewal Term | 0.423 | 0.783 | claude-opus-4-8 | -0.360 |
| Most Favored Nation | 0.300 | 0.667 | claude-opus-4-6 | -0.367 |
| Competitive Restriction Exception | 0.047 | 0.415 | claude-opus-4-8 | -0.368 |
| Notice Period To Terminate Renewal | 0.418 | 0.800 | gpt-5.2 | -0.382 |
| Liquidated Damages | 0.131 | 0.521 | claude-opus-4-6 | -0.389 |
| No-Solicit Of Employees | 0.366 | 0.767 | claude-opus-4-6 | -0.400 |
| Uncapped Liability | 0.150 | 0.552 | claude-opus-4-8 | -0.402 |
| Unlimited/All-You-Can-Eat-License | 0.083 | 0.520 | claude-opus-4-6 | -0.437 |
| Covenant Not To Sue | 0.107 | 0.642 | claude-opus-4-6 | -0.535 |
| Parties | 0.247 | 0.954 | claude-opus-4-8 | -0.707 |
| Source Code Escrow | 0.100 | 0.917 | gpt-4o | -0.817 |

**Pilot wins 2 categories, matches 0 (±2 pts). Overall gap: -27.0 AUPR pts.**

## Signal band (pre-committed thresholds): **RED**

*Presented for the human funding decision — not auto-acted on. A RED result is still a successful pilot outcome.*
