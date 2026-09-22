# Notebook 11 — Deep-Learning Results Summary

## Experimental comparison

Four deep-learning architectures—LSTM, GRU, BiLSTM, and TCN—were evaluated at 15-, 30-, and 60-minute forecast horizons. The principal ablation was the increment from **F3** to **F4**. F3 contained the continuous lunar and Sun–Moon astronomical controls in addition to the preceding history, meteorological, Gregorian, and solar-astronomy features, whereas F4 additionally introduced the categorical Panchang representation. Consequently, the F3→F4 comparison tests whether the categorical representation contributes predictive information beyond the corresponding continuous astronomical state.

All final TEST predictions were generated under the previously frozen architecture and epoch-selection protocol. Model selection and epoch selection used TRAIN/VALIDATION data only; TEST predictions were evaluated once without post-hoc clipping. Uncertainty for F3→F4 was estimated using a paired UTC forecast-valid calendar-day block bootstrap with 92 day blocks and 10,000 replicates. Daylight analyses used the pre-existing valid-time definition of solar elevation > 0°, yielding 17,851 daylight observations from 26,496 TEST samples.

## Core F3→F4 result

The Panchang increment did **not** produce a uniform effect across architectures or forecast horizons. Among the six horizon/scope conditions, only one showed unanimous point-estimate direction across all four architectures: the 60-minute all-row condition, where all four models had higher MAE under F4 than under F3. Even there, three model-specific bootstrap intervals were entirely below zero while the LSTM interval crossed zero. No horizon/scope condition had all four model-specific confidence intervals on the same side of zero.

The clearest positive cross-architecture pattern occurred at 30 minutes. For all TEST rows, LSTM improved by +5.80% (95% CI +3.55% to +7.94%), BiLSTM by +8.66% (+6.53% to +10.83%), and TCN by +30.93% (+27.94% to +33.98%). In contrast, GRU changed by -5.46% (-7.89% to -3.16%), giving a clearly opposite architecture-specific result. The same 3-versus-1 directional pattern remained in the daylight subset.

At 15 minutes the evidence was more heterogeneous: both the all-row and daylight analyses divided evenly between two positive and two negative point estimates. At 60 minutes the direction shifted toward degradation. All four all-row point estimates were negative, while the daylight analysis contained three negative and one positive result. Thus, restricting evaluation to physically relevant daylight periods did not remove the architecture dependence.

## Architecture-specific behavior

The architectures exhibited substantially different responses to the F3→F4 increment. GRU produced negative point estimates with confidence intervals entirely below zero in all six horizon/scope conditions. TCN showed the opposite direction at shorter horizons: its 15- and 30-minute all-row and daylight estimates were positive with intervals entirely above zero, whereas both 60-minute estimates were negative with intervals entirely below zero. BiLSTM showed positive evidence at 30 minutes but negative evidence at 60 minutes, with mixed results at 15 minutes. LSTM showed positive daylight estimates with intervals above zero at all three horizons, although its all-row 15- and 60-minute intervals crossed zero.

These divergent responses show that the categorical Panchang representation interacts strongly with model architecture and forecast horizon. The results therefore do not support a single architecture-independent claim that F4 is uniformly superior to F3, nor do they support a blanket conclusion that the Panchang representation has no predictive effect.

## Interpretation

The most defensible interpretation is that **the F3→F4 increment is architecture- and horizon-dependent**. In several model/horizon combinations, F4 reduced held-out MAE with bootstrap intervals entirely above zero; in others, it increased MAE with intervals entirely below zero. This bidirectional evidence is more consistent with a representation-dependent modeling effect than with a universal forecasting advantage.

Because F3 already controls for continuous lunar and Sun–Moon astronomical state, any F4 improvement should be interpreted specifically as predictive information associated with the **categorical Panchang representation beyond those continuous astronomical variables**. It should not be interpreted as evidence of a causal physical or Vedic-calendar mechanism.

## Cross-model evidence summary

Across the 24 architecture × horizon × scope comparisons, 11 point estimates favored F4 and 13 favored F3. The corresponding bootstrap intervals were entirely above zero in 10 comparisons, entirely below zero in 10, and crossed zero in 4. These counts are descriptive only; they are not treated as independent observations and are not combined into a pooled effect size.

Only **1 of 6** horizon/scope conditions had unanimous point-estimate direction across all four architectures, and **0 of 6** had all four bootstrap intervals on the same side of zero. Accordingly, cross-architecture effect-size averaging was intentionally avoided.

## Main conclusion

The deep-learning experiments provide evidence that Panchang categorical features can alter renewable-energy forecasting performance beyond continuous astronomical controls, but the direction and magnitude of that effect depend strongly on the neural architecture and forecast horizon. The 30-minute horizon provides the strongest repeated positive pattern across multiple architectures, whereas the 60-minute horizon provides the clearest evidence against a general benefit. These results support treating Panchang information as a potentially useful but model-dependent representation rather than as a universally beneficial forecasting feature.