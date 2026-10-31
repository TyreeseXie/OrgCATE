# Panel fixed-effects comparison

The fixed-effects benchmark expands each frozen firm-level replication
into six relative periods. Firm and period effects are absorbed, and the
treatment coefficient is estimated from `T_i x Post_t` with firm-clustered
CR1 uncertainty. The existing True ATE, DML-ATE, and CATE are read
from the frozen confirmation evidence.

The comparison contains `10` replications
for each of the nine one-dimensional HTE forms.

| HTE form | True ATE | Firm + period FE | DML-ATE | CATE | FE bias vs ATE | FE bias vs ATT |
|---|---:|---:|---:|---:|---:|---:|
| homogeneous | 1.0000 | 0.9967 | 1.0274 | 0.9824 | -0.0033 | -0.0033 |
| linear-monotone | 1.6104 | 1.8995 | 1.6298 | 1.5718 | 0.2895 | -0.0033 |
| nonlinear-monotone | 1.9220 | 2.1814 | 1.9429 | 1.8644 | 0.2598 | -0.0033 |
| steep-continuous | 1.4852 | 1.8708 | 1.5058 | 1.4514 | 0.3863 | -0.0033 |
| inverted-u | 0.9209 | 0.9910 | 0.9526 | 0.9078 | 0.0668 | -0.0033 |
| u-shaped | -0.9209 | -0.9975 | -0.8946 | -0.8302 | -0.0734 | -0.0033 |
| kinked | 2.2770 | 2.7763 | 2.2965 | 2.2093 | 0.5019 | -0.0033 |
| threshold-step | 1.2178 | 1.5006 | 1.2432 | 1.1880 | 0.2828 | -0.0033 |
| sign-reversal | 0.8994 | 0.4087 | 0.9250 | 0.8827 | -0.4913 | -0.0033 |

The ATT columns identify the estimand represented by the conventional
fixed-effects coefficient when treatment selection and effect
heterogeneity are associated. The DML and CATE values are unchanged
frozen outputs from the unified morphology workflow.
