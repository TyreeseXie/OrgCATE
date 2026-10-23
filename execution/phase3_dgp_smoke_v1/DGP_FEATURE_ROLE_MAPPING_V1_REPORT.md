# DGP Feature Role Mapping v1 — Evidence-based Freeze

Truth-safe pool: **8,528** firms from the fixed one-anchor-per-firm L3 USD backbone (only 1.39% removed for missing primary truth-role features).

| role   | feature                 | meaning                 |   missing_pct |   n_unique |          p01 |       median |       p99 | why                                                                                                          |
|:-------|:------------------------|:------------------------|--------------:|-----------:|-------------:|-------------:|----------:|:-------------------------------------------------------------------------------------------------------------|
| M1     | at_rank_year_currency   | relative firm size      |     0         |       8610 |   0.00865677 |  0.457946    |  0.984658 | 0% missing, continuous, strong support, intuitive moderator; sale/employment provide real correlated proxies |
| M2     | cash_at                 | financial slack         |     0         |       8477 |   0          |  0.157088    |  0.988943 | 0% missing, broad continuous support, only moderate correlation with size/leverage                           |
| M3     | intangibles_at          | intangibility           |     0.682239  |       5855 |   0          |  0.0498492   |  0.849286 | <1% missing, distinct organizational resource dimension                                                      |
| C1     | roa_ni                  | profitability           |     0.0231267 |       8643 | -36.1116     | -0.0396674   |  0.404101 | near-complete and economically relevant to treatment selection and outcomes                                  |
| C2     | debt_at                 | leverage                |     0.335338  |       7475 |   0          |  0.2202      | 17.2063   | near-complete, distinct financing condition                                                                  |
| C3     | ppe_at                  | capital intensity       |     0.0693802 |       8135 |   0          |  0.119138    |  0.942503 | near-complete, real operating structure                                                                      |
| C4     | log_at_slope_L          | size trajectory         |     0         |       8633 |  -0.861381   |  0.0484146   |  1.85363  | complete longitudinal state with modest correlations                                                         |
| P1     | capx_at                 | investment intensity    |     0.242831  |       7679 |   0          |  0.0178466   |  0.371225 | near-complete; prognostic-only in DGP truth                                                                  |
| P2     | roa_ni_slope_L          | profitability trend     |     0.0346901 |       8645 | -14.395      | -0.00179416  |  9.34654  | near-complete longitudinal predictor                                                                         |
| P3     | cash_at_slope_L         | slack trend             |     0         |       8582 |  -0.342868   | -0.000939525 |  0.335357 | complete and weakly correlated with level factors                                                            |
| A1     | debt_at_slope_L         | leverage trajectory     |     0.231267  |       7820 |  -3.00844    |  0           |  5.91636  | near-complete; assignment-only role in primary truth                                                         |
| R1     | sale_rank_year_currency | size proxy — sales      |     0.0231267 |       7549 |   0.048965   |  0.450797    |  0.984343 | highly correlated real proxy for M1                                                                          |
| R2     | asinh_emp               | size proxy — employment |     5.74699   |       2934 |   0          |  0.431486    |  5.74428  | highly correlated real proxy for M1; retained as analyst candidate with missingness                          |

## Key design logic
- M1 uses contemporaneous firm-size percentile; sales and employment are retained as real correlated proxies rather than independent truth drivers.
- M2 cash slack and M3 intangibility provide distinct moderator dimensions.
- C1–C4 are observed confounders entering both treatment and untreated outcome functions.
- P1–P3 enter untreated outcome only.
- A1 enters treatment assignment only.
- All truth-generation inputs are converted to frozen empirical percentile scores on the complete truth-safe backbone, except M1 which is already contemporaneous percentile.
