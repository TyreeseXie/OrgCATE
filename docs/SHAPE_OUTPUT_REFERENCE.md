# Functional-form output reference

For each supported result, `shape_quantities` retains estimates from the three
nuisance-model refits. `key_quantities` reports their final summary, normally
the median, and `interpretation_en` gives a one-sentence description.

CATE magnitudes use the outcome scale. Locations use the scale of the focal
characteristic; the benchmark uses a reference rank from 0 to 1.

| Functional form | Reported quantities |
|---|---|
| `homogeneous` | constant effect, global slope, and CATE mean/minimum/maximum/range |
| `linear-monotone` | slope, direction, and endpoint effects |
| `nonlinear-monotone` | transition center, amplitude, 25%-75% transition width, effective center gradient, direction, local slopes, endpoint effects, and raw `k` as a diagnostic |
| `steep-continuous` | transition center, amplitude, 25%-75% transition width, effective center gradient, direction, local slopes, endpoint effects, and raw `k` as a diagnostic |
| `inverted-u` | peak location, curvature, peak effect, tail effects, peak-to-tail contrasts, and slopes on both sides |
| `u-shaped` | trough location, curvature, trough effect, tail effects, trough-to-tail contrasts, and slopes on both sides |
| `kinked` | kink location and effect, left and right slopes, slope change, segment directions, and endpoint effects |
| `threshold-step` | number and locations of boundaries, effects within bands, and adjacent-band differences |
| `sign-reversal` | zero crossing, endpoint effects and signs, global slope, and slopes on both sides |

For `threshold-step`, boundary refers to the operational threshold-like form
used by the procedure and represents an operational candidate boundary with a
localized two-side effect contrast.
Multiple boundaries use the same array-valued fields and the
`multi-threshold-step` subtype.

For tanh-shaped transitions, amplitude and raw `k` can compensate while
producing almost the same substantive CATE curve. The operational distinction
between ordinary nonlinear and steep-continuous transitions therefore uses the
effective center gradient `|amplitude x k|`. The reported 25%-75% transition
width is `2 atanh(.5) / k` on the focal characteristic's scale.

If an earlier check fails, or if a required property is unsupported,
`key_quantities` is empty and `interpretation_en` records why the shape-specific
summary was not reported.
