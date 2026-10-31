# Optional LLM-Assisted Theory Development

## 1. How to use this file

This file serves two purposes:

1. The worked example shows how an estimated CATE morphology can be translated into a theory-development prompt.
2. The reusable template can be applied to a different treatment, outcome, population, focal characteristic, or morphology.

In the reusable template, every item that the researcher must replace is shown in **bold square brackets**. The text following “Example” demonstrates the level of specificity expected; it is not part of the general template.

The language model receives estimated results from the analysis workflow, not data-generating truth. DML supplies the average treatment effect, one multivariable causal forest supplies the focal heterogeneity signal and firm-level CATE estimates, orthogonal morphology reconstruction supplies the functional form and its parameters, and the untouched test sample supplies independent confirmation.

## 2. Worked example

### Substantive inputs

- **Treatment:** Formal deployment of an AI-enabled strategic decision-support system in 2021 (`1 = deployed`, `0 = not deployed`).
- **Outcome:** Standardized innovation productivity during 2022-2024, defined as quality-adjusted patent output relative to R&D expenditure.
- **Population and period:** U.S. publicly listed manufacturing and knowledge-intensive business-service firms observed from 2018 through 2024; 2018-2020 supplies pre-treatment firm characteristics.
- **Focal firm characteristic:** Relative position `u` in the pre-adoption total-asset distribution. `u=.10` represents relatively small firms, `u=.50` the middle, and `u=.90` relatively large firms.

### Estimated analysis output

| Analysis stage | Estimated output | Organizational interpretation |
|---|---:|---|
| DML | ATE = `.9246`; 95% CI = `[.8210, 1.0283]` | AI deployment increases the standardized outcome by about .925 units on average. |
| DML-ATE/CATE check | TRAIN-target CATE mean = `.9966`; standardized gap = `.0572`; threshold = `.10`; result = `PASS` | The two estimates target the same firms and are sufficiently aligned to continue the morphology analysis. |
| Causal forest | Relative firm-size importance = `.6573`; rank = `1 of 12` | Firm size is the leading heterogeneity signal in this analysis. |
| Test-sample CATE | Mean = `.8772`; SD = `.9876`; P10 = `-.7369`; median = `1.2560`; P90 = `1.9005` | The positive average effect conceals substantial cross-firm variation. |
| Reconstructed morphology | Inverted U: `tau_hat(u) = 1.9296 - 13.1243(u-.50)^2` | The curve rises and then falls; the quadratic coefficient is `-13.1243`. |
| Peak and curvature | Peak at `u=.50`; curvature = `-26.2485` | Estimated gains are greatest near the middle of the size distribution. |
| Refit stability | Curvatures = `-26.5643`, `-26.2485`, `-26.6378`; median = `-26.5643` | All three refits imply closely aligned concavity. |
| Point effects | At `u=.10/.25/.50/.75/.90`: `-.1703/1.1093/1.9296/1.1093/-.1703` | Effects rise from smaller firms to the middle and then decline. |
| Arm-specific quantities | Peak-minus-endpoint contrasts = `2.0999/2.0999`; average slopes below/above peak = `+5.2497/-5.2497` | Both the rising and declining arms are substantively large. |
| Independent confirmation | `3 of 3` refits pass test-sample confirmation; cross-refit curve agreement = `.999997` | The pattern remains stable across repeated fitting and held-out evidence. |

The function and its coefficients are outputs of orthogonal morphology reconstruction after the causal forest. They are not direct coefficients from the forest. Simulation truth, oracle accuracy metrics, and preassigned active-variable labels are excluded.

## 3. Reusable prompt template with example entries

Copy the block below. Replace each **bold bracketed field** while preserving the headings and task structure.

> ### Role
>
> Act as an Organization Science scholar specializing in **[RESEARCH DOMAIN. Example: artificial intelligence, organization design, and innovation]**. Use abductive theory development to explain an empirically estimated pattern of treatment-effect heterogeneity, compare plausible organizational mechanisms, and derive propositions that can be tested in subsequent research.
>
> ### Study setting
>
> - **Treatment:** **[DEFINE THE SUBSTANTIVE INTERVENTION, ITS TIMING, AND CODING. Example: formal deployment of an AI-enabled strategic decision-support system in 2021; 1 = deployed, 0 = not deployed]**
> - **Outcome:** **[DEFINE THE OUTCOME, MEASUREMENT WINDOW, AND UNIT. Example: standardized innovation productivity during 2022-2024, measured as quality-adjusted patent output relative to R&D expenditure]**
> - **Population and period:** **[DEFINE INDUSTRIES, COUNTRY OR REGION, SAMPLE, AND YEARS. Example: U.S. publicly listed manufacturing and knowledge-intensive business-service firms observed from 2018 through 2024, with 2018-2020 as the pre-treatment period]**
> - **Focal characteristic:** **[DEFINE THE ORGANIZATIONAL CHARACTERISTIC AND ITS SCALE. Example: relative pre-adoption firm size `u`, measured by position in the total-asset distribution]**
>
> ### Estimated analytical result
>
> - **DML average effect:** **[ENTER THE ESTIMATED ATE, STANDARD ERROR OR CONFIDENCE INTERVAL, AND OUTCOME UNIT. Example: ATE = .9246 standardized units; 95% CI [.8210, 1.0283]]**
> - **DML-ATE/CATE check:** **[ENTER THE TRAIN-TARGET CATE MEAN, STANDARDIZED GAP, PRE-SPECIFIED THRESHOLD, AND RESULT. Example: TRAIN-target CATE mean = .9966; standardized gap = .0572; threshold = .10; result = PASS]**
> - **Causal-forest signal:** **[ENTER THE FOCAL CHARACTERISTIC, ITS IMPORTANCE OR RANK, AND THE NUMBER OF CANDIDATE CHARACTERISTICS. Example: relative firm size ranks 1st among 12 characteristics; importance = .6573]**
> - **Held-out CATE distribution:** **[ENTER THE MEAN, DISPERSION, AND RELEVANT QUANTILES. Example: mean = .8772, SD = .9876, P10 = -.7369, median = 1.2560, P90 = 1.9005]**
> - **Selected morphology:** **[ENTER THE ESTIMATED FUNCTIONAL FORM AND EQUATION. Example: inverted U, `tau_hat(u) = 1.9296 - 13.1243(u-.50)^2`]**
> - **Key morphology parameters:** **[ENTER ALL PARAMETERS NEEDED TO INTERPRET THIS SHAPE. Depending on the morphology, these may include slopes, curvature, turning point, zero crossing, kink location, threshold location, band effects, or two-side contrasts. Example: peak at `u=.50`, curvature = `-26.2485`, peak-minus-endpoint contrasts = `2.0999/2.0999`, average slopes below/above the peak = `+5.2497/-5.2497`]**
> - **Representative CATE values:** **[ENTER ESTIMATED EFFECTS AT SUBSTANTIVELY MEANINGFUL POSITIONS. Example: at `u=.10/.25/.50/.75/.90`, CATE = `-.1703/1.1093/1.9296/1.1093/-.1703`]**
> - **Stability and independent confirmation:** **[ENTER REFIT STABILITY AND HELD-OUT CONFIRMATION RESULTS. Example: 3 of 3 refits pass test-sample confirmation; cross-refit curve agreement = .999997]**
>
> Use the theory-development tasks only when the DML-ATE/CATE check passes and the morphology result is supported by the analysis workflow. Treat these estimated quantities as the empirical phenomenon to explain. Keep the statistical pattern separate from the theoretical mechanisms proposed to explain it. Variable importance identifies a focal characteristic; it does not itself establish an organizational mechanism.
>
> ### Research question
>
> **[STATE THE ORGANIZATIONAL QUESTION IMPLIED BY THE ESTIMATED MORPHOLOGY. Example: Why does deployment of an AI-enabled strategic decision-support system produce its largest innovation-productivity gains among medium-sized firms, with weaker or negative gains toward both ends of the firm-size distribution?]**
>
> ### Tasks
>
> 1. Define the theoretical phenomenon in 150-200 words. Explain what is organizationally consequential about the difference between the average treatment effect and the estimated CATE pattern. Keep statistical evidence and theoretical interpretation distinct.
> 2. Search peer-reviewed literature published since **[LITERATURE START YEAR. Example: 2018]**, adding a small number of foundational studies when necessary. Prioritize *Organization Science*, *Academy of Management Journal*, *Academy of Management Review*, *Administrative Science Quarterly*, *Strategic Management Journal*, *Management Science*, and other UTD journals or authoritative journals in organization and management research.
> 3. Develop four to six theoretically distinguishable organizational mechanisms that could produce the estimated morphology. For each mechanism, specify the focal constructs, causal chain, explanation for each segment or region of the curve, observable mediator, boundary conditions, and at least one prediction that distinguishes it from the alternatives.
> 4. Select the two mechanisms that most completely explain the estimated pattern. State whether they compete, complement one another, or operate at different levels of analysis. Translate them into two to four testable propositions. Each proposition must identify the treatment, outcome, relevant range or region of the focal characteristic, expected direction, and theoretical mechanism.
> 5. Propose follow-up evidence that can distinguish the selected mechanisms. Include at least one mediation test, one additional boundary condition, and organizational- or individual-level variables that should be collected.
> 6. Return four sections: **Phenomenon statement**, **Mechanism-comparison table**, **Propositions**, and **Next evidence**. The comparison table must include theoretical lens, focal constructs, shape-specific explanatory logic, key literature, observable prediction, and data needed.
> 7. For every reference, provide authors, year, title, journal, DOI, and publisher webpage. Verify every source before including it and omit references that cannot be verified.

## 4. Suggested output structure

### A. Phenomenon statement

A concise account of the average effect, heterogeneous pattern, focal organizational characteristic, and theoretical puzzle.

### B. Mechanism-comparison table

| Theoretical lens | Focal constructs | Why it produces the estimated shape | Mediator | Boundary condition | Distinguishing prediction | Key sources |
|---|---|---|---|---|---|---|

### C. Propositions

Two to four propositions that connect the treatment, outcome, focal characteristic, estimated region of the curve, and organizational mechanism.

### D. Next evidence

The empirical evidence needed to adjudicate between the leading explanations.

## 5. Execution record

Use a multi-source research model such as ChatGPT Deep Research. Record the visible model name, run date, complete prompt, complete response, and source list. The researcher remains responsible for checking the literature, judging theoretical coherence, and deciding which explanation merits formal development and testing.
