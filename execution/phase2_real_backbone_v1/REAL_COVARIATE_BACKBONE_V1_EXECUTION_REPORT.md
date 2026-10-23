# Real Covariate Backbone v1 — Execution Report

**Verdict: PASS.** L1/L3/L5 registries and fixed one-anchor-per-firm L3 headline backbone constructed.

## Sample composition

|   L |   firm_anchor_rows |   unique_firms |   USD_firm_anchor_rows |   USD_unique_firms |
|----:|-------------------:|---------------:|-----------------------:|-------------------:|
|   1 |              87554 |          11611 |                  82084 |              10630 |
|   3 |              64090 |           9385 |                  60895 |               8648 |
|   5 |              45862 |           7238 |                  44256 |               6788 |

## Headline L3 anchor distribution

|   candidate_t0 |   firms |
|---------------:|--------:|
|           2013 |     929 |
|           2014 |     740 |
|           2015 |     647 |
|           2016 |     597 |
|           2017 |     475 |
|           2018 |     493 |
|           2019 |     471 |
|           2020 |     457 |
|           2021 |     428 |
|           2022 |     523 |
|           2023 |     619 |
|           2024 |     659 |
|           2025 |     741 |
|           2026 |     869 |

## Highest-missingness features

| feature              |   missing_pct |   n_unique_nonmissing |          p01 |      median |       p99 |
|:---------------------|--------------:|----------------------:|-------------:|------------:|----------:|
| xrd_at_volatility_L  |     37.6966   |                  4732 |   0          | 0.0137671   |  3.75617  |
| xrd_at_slope_L       |     37.6966   |                  4727 |  -1.66211    | 0           |  0.810465 |
| xrd_at               |     37.6503   |                  4659 |   0          | 0.0687343   |  2.8658   |
| xrd_at_mean_L        |     36.2165   |                  4811 |   0          | 0.0746759   |  3.56793  |
| xsga_at              |      9.1235   |                  7852 |   0.00778606 | 0.261114    | 23.4943   |
| asinh_emp            |      5.74699  |                  2934 |   0          | 0.431486    |  5.74428  |
| intangibles_at       |      0.682239 |                  5855 |   0          | 0.0498492   |  0.849286 |
| debt_at              |      0.335338 |                  7475 |   0          | 0.2202      | 17.2063   |
| goodwill_at          |      0.335338 |                  4322 |   0          | 0.000203803 |  0.596498 |
| capx_at              |      0.242831 |                  7679 |   0          | 0.0178466   |  0.371225 |
| debt_at_volatility_L |      0.231267 |                  7868 |   0          | 0.046433    | 20.7723   |
| debt_at_slope_L      |      0.231267 |                  7820 |  -3.00844    | 0           |  5.91636  |
| op_income_at         |      0.219704 |                  8619 | -24.8347     | 0.020194    |  0.384646 |
| equity_at            |      0.185014 |                  8621 | -71.0775     | 0.444821    |  0.974965 |
| cashflow_at          |      0.161887 |                  8623 | -11          | 0.0176455   |  0.341804 |

## Notes
- One GVKEY appears once in headline data.
- Anchor choice is deterministic hashed pseudo-random selection, not latest-year selection.
- Current/header metadata are absent.
- Missing XRD is preserved.
- Same-year x currency ECDF ranks are used for AT and SALE.
- Parquet serialization is deferred because the runtime lacks a parquet engine.
