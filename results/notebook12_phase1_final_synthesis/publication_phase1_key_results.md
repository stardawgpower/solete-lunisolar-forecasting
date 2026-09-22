| result_id | result | summary |
| --- | --- | --- |
| R1 | Gregorian increment F0 -> F1 | Lower MAE in 26 of 30 model-horizon conditions. |
| R2 | Solar-astronomy increment F1 -> F2 | Lower MAE in 17 of 30 model-horizon conditions. |
| R3 | Continuous lunar / Sun-Moon increment F2 -> F3 | Lower MAE in 7 of 30 conditions and higher MAE in 23 of 30 conditions. |
| R4 | Categorical Panchang increment F3 -> F4 | Across all-row TEST results, F4 lowers MAE in 17 of 30 model-horizon conditions and raises MAE in 13 of 30. |
| R5 | Unified F3 -> F4 bootstrap evidence | Across 60 model-horizon-scope comparisons, 12 CIs lie above zero, 16 below zero, and 32 cross zero. |
| R6 | 30-minute F3 -> F4 profile | Both all-row and daylight evaluations contain four CIs above zero, two below zero, and four crossing zero. |
| R7 | 60-minute F3 -> F4 profile | Resolved negative effects are more common than resolved positive effects in both evaluation scopes. |
| R8 | Model dependence | Ridge and GRU have all six CIs below zero; BiLSTM and TCN exhibit resolved positive and negative conditions. |
| R9 | All-row versus daylight robustness | Point direction agrees in 27 of 30 model-horizon pairs. |
| R10 | Scientific interpretation | F3 -> F4 effects are model- and horizon-dependent; the experiment supports predictive representation claims, not physical causal claims. |
