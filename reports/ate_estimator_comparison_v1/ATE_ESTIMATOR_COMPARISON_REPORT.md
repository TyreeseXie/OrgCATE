# Conventional constant-effect OLS and DML-ATE comparison

This diagnostic precedes CATE morphology analysis. Each of the nine
one-dimensional HTE generators is evaluated separately. The conventional
estimator is the single treatment coefficient from `Y ~ 1 + T + X`, with
all 12 pre-treatment characteristics and an HC3 standard error. It is called
the conventional benchmark because each firm contributes one treatment and
one outcome observation. HC3 standard errors quantify its uncertainty.

The comparison reports bias, RMSE, and interval coverage for both estimators.
Simulation truth is opened only in the evaluator report for these calculations.

Sample size is `3000` with an 80/20 firm split; results
use the TRAIN partition and `20` replications
per situation.

`ATE_ESTIMATOR_COMPARISON.svg` compares mean true ATE with mean OLS
and DML estimates across the nine HTE forms.

| HTE form | OLS bias | OLS RMSE | OLS coverage | DML bias | DML RMSE | DML coverage | Median relative OLS-DML deviation |
|---|---:|---:|---:|---:|---:|---:|---:|
| homogeneous | 0.0498 | 0.0621 | 70.0% | -0.0139 | 0.0557 | 95.0% | 7.0% |
| linear-monotone | 0.2642 | 0.2659 | 0.0% | -0.0169 | 0.0533 | 90.0% | 16.9% |
| nonlinear-monotone | 0.2425 | 0.2440 | 0.0% | 0.0068 | 0.0469 | 100.0% | 12.2% |
| steep-continuous | 0.3220 | 0.3236 | 0.0% | -0.0074 | 0.0520 | 100.0% | 22.7% |
| inverted-u | 0.1976 | 0.2037 | 5.0% | -0.0098 | 0.0440 | 100.0% | 21.5% |
| u-shaped | -0.0728 | 0.0870 | 75.0% | -0.0075 | 0.0504 | 100.0% | 8.7% |
| kinked | 0.3469 | 0.3507 | 0.0% | -0.0236 | 0.0527 | 95.0% | 16.7% |
| threshold-step | 0.2474 | 0.2505 | 0.0% | -0.0013 | 0.0347 | 100.0% | 21.4% |
| sign-reversal | -0.2756 | 0.2782 | 0.0% | -0.0192 | 0.0498 | 100.0% | 29.5% |

This comparison establishes the average-effect reference used before the
causal-forest and orthogonal-reconstruction stages estimate HTE morphology.
