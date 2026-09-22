# Phase-1 Final Synthesis

## Experimental scope

Phase 1 evaluated ten frozen forecasting models across three forecast horizons (15, 30, and 60 minutes) and five nested feature families (F0-F4). The held-out TEST period contained 26,496 samples per model-horizon-feature experiment. The central novelty comparison was F3 -> F4: the addition of categorical Panchang features after continuous solar, lunar, and Sun-Moon astronomical controls had already been included.

The five feature families were interpreted as follows:

- F0: forecast-history / meteorological baseline representation.
- F1: F0 plus Gregorian calendar features.
- F2: F1 plus solar astronomical features.
- F3: F2 plus continuous lunar / Sun-Moon astronomical features.
- F4: F3 plus categorical Panchang representation.

All analyses were chronological and the final TEST set was not used for new model selection or hyperparameter tuning in Notebook 12.

## Full feature-ablation pattern

Across the 30 model-horizon conditions, the F0 -> F1 Gregorian increment reduced TEST MAE in 26 conditions and increased it in 4. The F1 -> F2 solar-astronomy increment reduced MAE in 17 conditions and increased it in 13. The F2 -> F3 continuous lunar / Sun-Moon increment reduced MAE in 7 conditions and increased it in 23. The final F3 -> F4 categorical Panchang increment reduced MAE in 17 conditions and increased it in 13.

These counts describe the direction of the within-model ablation changes. They are not pooled effect-size estimates and are not used to rank models.

## Core F3 -> F4 result

The F3 -> F4 comparison was evaluated for both all TEST rows and daylight-only TEST rows, producing 60 model-horizon-scope comparisons. Positive values denote lower MAE for F4 relative to F3.

Across the 60 comparisons, 35 point estimates favored F4 and 25 favored F3. Paired UTC calendar-day block bootstrap intervals were constructed using 92 daily blocks and 10,000 bootstrap replicates. Twelve 95% intervals lay entirely above zero, sixteen lay entirely below zero, and thirty-two crossed zero.

The result therefore does not support a universal F4 benefit, a universal F4 degradation, or a claim that F3 and F4 are indistinguishable in every evaluated condition. Instead, the incremental effect is heterogeneous.

## Horizon dependence

The 30-minute horizon showed the most consistently favorable F3 -> F4 profile among the evaluated horizons. In both all-row and daylight evaluation, seven of ten point estimates favored F4; four confidence intervals lay entirely above zero, two lay entirely below zero, and four crossed zero.

At 60 minutes, the resolved pattern shifted in the opposite direction. In all-row evaluation, no confidence interval lay entirely above zero while four lay entirely below zero. In daylight evaluation, one interval lay above zero and four lay below zero.

This demonstrates that the incremental F3 -> F4 behavior cannot be summarized independently of forecast horizon.

## Model and architecture dependence

Substantial model dependence was observed.

Ridge and GRU produced negative F3 -> F4 point estimates in all six horizon-scope conditions, and all six of their corresponding confidence intervals lay below zero.

HGBR and XGBoost produced positive F3 -> F4 point estimates in all six conditions, but all six confidence intervals crossed zero.

LightGBM produced positive point estimates in all six conditions, with resolved positive intervals at the 30-minute horizon in both evaluation scopes.

LSTM contained resolved positive F3 -> F4 effects, particularly in daylight evaluation, but did not show the same direction in every condition.

BiLSTM exhibited resolved positive effects at 30 minutes and resolved negative effects at 60 minutes.

TCN showed the strongest horizon reversal: F4 reduced MAE relative to F3 at 15 and 30 minutes in both scopes, while F4 increased MAE relative to F3 at 60 minutes in both scopes.

These results demonstrate that the predictive consequence of adding the categorical Panchang representation depends strongly on the forecasting architecture and horizon.

## All-row versus daylight evaluation

The qualitative F3 -> F4 point-estimate direction agreed between all-row and daylight evaluation in 27 of 30 model-horizon pairs. Confidence-interval zero classification agreed in 28 of 30 pairs.

Daylight restriction therefore did not eliminate the heterogeneous pattern. It modified several architecture-specific results but preserved the broader conclusion that both favorable and unfavorable F3 -> F4 effects occur.

## Scientific interpretation

The Phase-1 evidence supports a predictive-representation interpretation only. It shows that categorical Panchang variables can alter out-of-sample forecasting behavior after continuous astronomical controls are included, but the direction and magnitude are model- and horizon-dependent.

The experiment does not establish that Panchang categories physically cause changes in photovoltaic generation. It also does not justify treating F4 as the globally superior feature representation whenever F4 improves over F3, because the F3 -> F4 comparison is incremental and other feature families or persistence baselines may still have lower absolute error.

No cross-architecture effect-size average is used as the main scientific conclusion. The central finding is the structured heterogeneity of the F3 -> F4 effect.

## Phase-1 conclusion

Phase 1 establishes that the categorical Panchang increment is neither universally beneficial nor universally detrimental beyond continuous solar/lunar astronomical controls. Instead, its predictive contribution is conditional on model architecture, forecast horizon, and, in a smaller number of cases, evaluation scope.

This architecture- and horizon-dependent behavior is the principal empirical result to carry forward into the manuscript and any future Phase-2 validation.
