# Final Results Synthesis

## Primary F3 vs F4 evidence

Across the 24 matched model × horizon × scope
conditions available in both datasets, the exact evidence
classification agreed in 13/24
conditions. The same resolved direction was observed in
9/24 conditions.

Ridge showed the clearest cross-dataset replication:
F3 had lower MAE in all six evaluated conditions in both
SOLETE and PVOD.

LightGBM showed weaker transfer of the SOLETE pattern.
SOLETE contained a mixture of F3-lower, F4-lower, and
unresolved comparisons, whereas all six PVOD LightGBM
comparisons had 95% confidence intervals crossing zero.

LSTM remained context-dependent across dataset, horizon,
and evaluation scope.

TCN showed the strongest external F4 tendency in PVOD,
where F4 had lower MAE in 5 of 6 conditions. However,
the corresponding SOLETE pattern was more heterogeneous,
indicating that the effect is not a universal
condition-independent advantage.

## Coding sensitivity

SOLETE showed resolved sensitivity to reference coding in
some conditions, but PVOD did not: all 24 PVOD
F4-real-versus-reference comparisons had 95% confidence
intervals crossing zero.

Therefore, coding sensitivity was not replicated as a
stable cross-dataset effect.

## Alignment controls

For SOLETE alignment-null comparisons:

- aligned F4_real lower MAE: 35/144
- shifted/permuted control lower MAE: 58/144
- unresolved: 51/144

For PVOD alignment-null comparisons:

- aligned F4_real lower MAE: 9/144
- shifted/permuted control lower MAE: 42/144
- unresolved: 93/144

Thus, the correctly aligned Panchang representation did not
systematically dominate shifted or permuted categorical
controls in either dataset.

This weakens an interpretation in which forecasting gains
arise primarily from uniquely correct Panchang temporal
alignment.

## Component interaction

SOLETE component analysis showed strong architecture
dependence in non-additivity.

Mean absolute full-F4-minus-summed-component discrepancy:

- Ridge: 0.001035 kW
- LightGBM: 0.003848 kW
- LSTM: 0.028157 kW
- TCN: 0.174889 kW

The much larger discrepancies for neural architectures,
especially TCN, indicate that the full categorical
representation cannot generally be interpreted as the
simple additive sum of isolated Panchang-family effects.

## Final interpretation

Across SOLETE and PVOD, categorical Panchang features behave
primarily as an architecture-dependent representation of
temporal and astronomical state.

Their forecasting effect depends on model inductive bias,
forecast horizon, dataset, coding convention, temporal
alignment, and interactions among categorical feature
families.

The evidence does not establish a causal or physical effect
of Panchang categories on photovoltaic generation.
