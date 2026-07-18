# nightwing, every model, every category (official CUAD AUPR, full 102-contract test split)

## Overall

| Model | AUPR | P@80R | P@90R | gap to best frontier |
|---|---|---|---|---|
| claude-opus-4-8 | **0.561** | 0.000 | 0.000 | +0.000 |
| claude-opus-4-6 | **0.498** | 0.000 | 0.000 | -0.063 |
| gpt-5.2 | **0.423** | 0.000 | 0.000 | -0.138 |
| gpt-4o | **0.421** | 0.000 | 0.000 | -0.140 |
| nightwing-v2-14b (extractive) | **0.389** | 0.000 | 0.000 | -0.173 |
| nightwing-v1-14b (generative) | **0.291** | 0.000 | 0.000 | -0.270 |

## Per-category, the full grid (40 scorable; 'Price Restrictions' excluded, 0 positives in test)

Best score per category in **bold**.

| Category | claude-opus-4-8 | claude-opus-4-6 | gpt-5.2 | gpt-4o | nightwing-v2-14b (extractive) | nightwing-v1-14b (generative) |
|---|---|---|---|---|---|---|
| Affiliate License-Licensee | 0.350 | **0.557** | 0.438 | 0.251 | 0.286 | 0.398 |
| Affiliate License-Licensor | **0.405** | 0.278 | 0.123 | 0.076 | 0.379 | 0.210 |
| Agreement Date | 0.117 | 0.168 | 0.054 | 0.079 | **0.829** | 0.687 |
| Anti-Assignment | **0.581** | 0.556 | 0.440 | 0.430 | 0.497 | 0.224 |
| Audit Rights | **0.441** | 0.265 | 0.255 | 0.207 | 0.307 | 0.147 |
| Cap On Liability | **0.523** | 0.419 | 0.333 | 0.321 | 0.359 | 0.194 |
| Change Of Control | 0.337 | 0.223 | 0.143 | **0.371** | 0.216 | 0.164 |
| Competitive Restriction Exception | **0.415** | 0.335 | 0.225 | 0.181 | 0.286 | 0.047 |
| Covenant Not To Sue | 0.606 | **0.642** | 0.467 | 0.213 | 0.539 | 0.107 |
| Document Name | 0.887 | 0.839 | 0.471 | 0.800 | **0.890** | 0.711 |
| Effective Date | 0.084 | 0.105 | 0.070 | 0.154 | **0.395** | 0.369 |
| Exclusivity | 0.296 | **0.465** | 0.410 | 0.361 | 0.368 | 0.215 |
| Expiration Date | 0.708 | 0.676 | 0.589 | 0.643 | **0.853** | 0.633 |
| Governing Law | **0.924** | 0.823 | 0.899 | 0.849 | 0.889 | 0.669 |
| Insurance | 0.341 | **0.353** | 0.204 | 0.298 | 0.252 | 0.226 |
| Ip Ownership Assignment | 0.309 | 0.280 | **0.351** | 0.234 | 0.167 | 0.143 |
| Irrevocable Or Perpetual License | **0.659** | 0.616 | 0.598 | 0.579 | 0.543 | 0.594 |
| Joint Ip Ownership | **0.524** | **0.524** | 0.231 | 0.489 | 0.500 | 0.184 |
| License Grant | **0.528** | 0.485 | 0.428 | 0.334 | 0.363 | 0.365 |
| Liquidated Damages | 0.502 | **0.521** | 0.318 | 0.483 | 0.236 | 0.131 |
| Minimum Commitment | 0.172 | **0.257** | 0.157 | 0.189 | 0.178 | 0.162 |
| Most Favored Nation | **0.667** | **0.667** | **0.667** | 0.333 | 0.200 | 0.300 |
| No-Solicit Of Customers | 0.501 | **0.573** | 0.289 | 0.300 | 0.399 | 0.296 |
| No-Solicit Of Employees | 0.661 | **0.767** | 0.626 | 0.692 | 0.446 | 0.366 |
| Non-Compete | **0.517** | 0.386 | 0.236 | 0.179 | 0.293 | 0.164 |
| Non-Disparagement | 0.369 | 0.393 | 0.343 | **0.396** | 0.208 | 0.229 |
| Non-Transferable License | **0.503** | 0.488 | 0.447 | 0.285 | 0.196 | 0.442 |
| Notice Period To Terminate Renewal | 0.681 | 0.677 | **0.800** | 0.757 | 0.607 | 0.418 |
| Parties | **0.954** | 0.926 | 0.892 | 0.853 | 0.326 | 0.247 |
| Post-Termination Services | 0.103 | 0.073 | 0.035 | 0.050 | **0.106** | 0.053 |
| Renewal Term | **0.783** | 0.691 | 0.593 | 0.608 | 0.731 | 0.423 |
| Revenue/Profit Sharing | **0.423** | 0.412 | 0.280 | 0.269 | 0.353 | 0.181 |
| Rofr/Rofo/Rofn | **0.411** | 0.356 | 0.372 | 0.228 | 0.143 | 0.213 |
| Source Code Escrow | 0.600 | 0.800 | 0.800 | **0.917** | 0.200 | 0.100 |
| Termination For Convenience | **0.618** | 0.502 | 0.488 | 0.404 | 0.472 | 0.463 |
| Third Party Beneficiary | **0.695** | 0.521 | 0.355 | 0.102 | 0.199 | 0.585 |
| Uncapped Liability | **0.552** | 0.359 | 0.349 | 0.063 | 0.415 | 0.150 |
| Unlimited/All-You-Can-Eat-License | 0.333 | **0.520** | 0.147 | 0.100 | 0.177 | 0.083 |
| Volume Restriction | 0.014 | 0.017 | 0.038 | 0.010 | **0.102** | 0.008 |
| Warranty Duration | **0.404** | 0.368 | 0.345 | 0.262 | 0.214 | 0.053 |

## Head-to-head: categories won by each specialist

| Specialist | vs claude-opus-4-6 | vs claude-opus-4-8 | vs gpt-4o | vs gpt-5.2 | beats ALL frontier |
|---|---|---|---|---|---|
| nightwing-v1-14b (generative) | 3/40 | 3/40 | 11/40 | 11/40 | 2 |
| nightwing-v2-14b (extractive) | 11/40 | 8/40 | 25/40 | 22/40 | 6 |

## Signal band (pre-committed thresholds, v1 pilot decision): **RED**

*v1 gap -27.0 pts, 2 outright wins. Presented for the human funding decision, not auto-acted on. A RED result is still a successful pilot outcome. v2 (extractive framing, same budget) closed the gap to -17.3 pts, see docs/RUN_JOURNAL_V2.md.*
