# CUAD per-category frontier weakness (P1.6)

Overall AUPR, **claude-opus-4-8** 0.561 | **claude-opus-4-6** 0.498 | **gpt-5.2** 0.423

40 categories scored ( 'Price Restrictions' excluded: 0 positives in test ).

## Top-10 winnable categories (weakest frontier, most room for the pilot)

| # | Category | best frontier AUPR | (by) | mean |
|---|---|---|---|---|
| 1 | Volume Restriction | 0.038 | gpt-5.2 | 0.023 |
| 2 | Post-Termination Services | 0.103 | claude-opus-4-8 | 0.071 |
| 3 | Effective Date | 0.105 | claude-opus-4-6 | 0.086 |
| 4 | Agreement Date | 0.168 | claude-opus-4-6 | 0.113 |
| 5 | Minimum Commitment | 0.257 | claude-opus-4-6 | 0.195 |
| 6 | Change Of Control | 0.337 | claude-opus-4-8 | 0.234 |
| 7 | Ip Ownership Assignment | 0.351 | gpt-5.2 | 0.313 |
| 8 | Insurance | 0.353 | claude-opus-4-6 | 0.299 |
| 9 | Non-Disparagement | 0.393 | claude-opus-4-6 | 0.368 |
| 10 | Warranty Duration | 0.404 | claude-opus-4-8 | 0.372 |

## Full 41-category table (sorted weakest-frontier first)

| Category | claude-opus-4-8 | claude-opus-4-6 | gpt-5.2 | best | mean |
|---|---|---|---|---|---|
| Volume Restriction | 0.014 | 0.017 | 0.038 | **0.038** | 0.023 |
| Post-Termination Services | 0.103 | 0.073 | 0.035 | **0.103** | 0.071 |
| Effective Date | 0.084 | 0.105 | 0.070 | **0.105** | 0.086 |
| Agreement Date | 0.117 | 0.168 | 0.054 | **0.168** | 0.113 |
| Minimum Commitment | 0.172 | 0.257 | 0.157 | **0.257** | 0.195 |
| Change Of Control | 0.337 | 0.223 | 0.143 | **0.337** | 0.234 |
| Ip Ownership Assignment | 0.309 | 0.280 | 0.351 | **0.351** | 0.313 |
| Insurance | 0.341 | 0.353 | 0.204 | **0.353** | 0.299 |
| Non-Disparagement | 0.369 | 0.393 | 0.343 | **0.393** | 0.368 |
| Warranty Duration | 0.404 | 0.368 | 0.345 | **0.404** | 0.372 |
| Affiliate License-Licensor | 0.405 | 0.278 | 0.123 | **0.405** | 0.269 |
| Rofr/Rofo/Rofn | 0.411 | 0.356 | 0.372 | **0.411** | 0.380 |
| Competitive Restriction Exception | 0.415 | 0.335 | 0.225 | **0.415** | 0.325 |
| Revenue/Profit Sharing | 0.423 | 0.412 | 0.280 | **0.423** | 0.372 |
| Audit Rights | 0.441 | 0.265 | 0.255 | **0.441** | 0.321 |
| Exclusivity | 0.296 | 0.465 | 0.410 | **0.465** | 0.390 |
| Non-Transferable License | 0.503 | 0.488 | 0.447 | **0.503** | 0.480 |
| Non-Compete | 0.517 | 0.386 | 0.236 | **0.517** | 0.380 |
| Unlimited/All-You-Can-Eat-License | 0.333 | 0.520 | 0.147 | **0.520** | 0.333 |
| Liquidated Damages | 0.502 | 0.521 | 0.318 | **0.521** | 0.447 |
| Cap On Liability | 0.523 | 0.419 | 0.333 | **0.523** | 0.425 |
| Joint Ip Ownership | 0.524 | 0.524 | 0.231 | **0.524** | 0.426 |
| License Grant | 0.528 | 0.485 | 0.428 | **0.528** | 0.480 |
| Uncapped Liability | 0.552 | 0.359 | 0.349 | **0.552** | 0.420 |
| Affiliate License-Licensee | 0.350 | 0.557 | 0.438 | **0.557** | 0.449 |
| No-Solicit Of Customers | 0.501 | 0.573 | 0.289 | **0.573** | 0.455 |
| Anti-Assignment | 0.581 | 0.556 | 0.440 | **0.581** | 0.526 |
| Termination For Convenience | 0.618 | 0.502 | 0.488 | **0.618** | 0.536 |
| Covenant Not To Sue | 0.606 | 0.642 | 0.467 | **0.642** | 0.572 |
| Irrevocable Or Perpetual License | 0.659 | 0.616 | 0.598 | **0.659** | 0.624 |
| Most Favored Nation | 0.667 | 0.667 | 0.667 | **0.667** | 0.667 |
| Third Party Beneficiary | 0.695 | 0.521 | 0.355 | **0.695** | 0.524 |
| Expiration Date | 0.708 | 0.676 | 0.589 | **0.708** | 0.658 |
| No-Solicit Of Employees | 0.661 | 0.767 | 0.626 | **0.767** | 0.684 |
| Renewal Term | 0.783 | 0.691 | 0.593 | **0.783** | 0.689 |
| Notice Period To Terminate Renewal | 0.681 | 0.677 | 0.800 | **0.800** | 0.719 |
| Source Code Escrow | 0.600 | 0.800 | 0.800 | **0.800** | 0.733 |
| Document Name | 0.887 | 0.839 | 0.471 | **0.887** | 0.732 |
| Governing Law | 0.924 | 0.823 | 0.899 | **0.924** | 0.882 |
| Parties | 0.954 | 0.926 | 0.892 | **0.954** | 0.924 |
