# ADNI Analysis Conventions (prespecified)

For the finalized Paper A pre-registration-style conventions, see `analysis_conventions_v1.0.md`.

This document locks primary definitions for cohort construction, biomarker positivity, gray-zone handling, and statistical reporting for Paper A (paired benchmark dependence + gray-zone workflow). Sensitivity analyses are listed explicitly.

## Paper scope

- **Primary (Paper A):** paired comparator swap (PET vs CSF benchmark) and gray-zone/reflex-testing workflow
- **Secondary (Paper B/C):** latent class / discordance biology (to be specified after Paper A triad table is validated)

## Cohort construction (Triad cohort)

**Inclusion**
- Participant has plasma, amyloid PET, and CSF measures meeting the time-window rule below.
- Required identifiers: `RID` (participant), modality dates, modality-specific values (continuous + derived positivity).

**Index triad selection (one row per participant, primary)**
- Candidate triads are all plasma×PET×CSF combinations that pass the time-window rule.
- Select the *nearest-in-time triad* by minimizing:
  1) the maximum pairwise |Δdays| across the three modalities, then
  2) the sum of pairwise |Δdays|, then
  3) earliest median date (tie-breaker).

**Primary time-window rule (TBD final values)**
- Default acceptance constraints:
  - `|plasma_date - PET_date| <= 90` days
  - `|plasma_date - CSF_date| <= 90` days
  - `|PET_date - CSF_date| <= 180` days

**Sensitivity windows**
- Tight: 60/60/120 days (plasma–PET / plasma–CSF / PET–CSF)
- Wide: 120/120/240 days

**Repeated measures**
- Primary uses one triad per participant (nearest-in-time).
- Sensitivity: allow multiple non-overlapping triads per participant (requires explicit correlation handling).

## Benchmark / positivity definitions

These must be locked per modality (primary + alternates). Continuous values are retained in the triad table even when binarized for performance metrics.

### Amyloid PET (benchmark option 1)
- **Primary PET metric (TBD):** Centiloid vs tracer-specific SUVR.
- **Tracer handling:** include tracer indicator; define whether to:
  - harmonize via Centiloids, or
  - use tracer-specific SUVR thresholds and report stratified + pooled results.
- **Primary PET positivity rule (TBD):** explicit threshold + source citation.
- **Alternates:** at least one threshold alternate (reviewer sensitivity).

### CSF amyloid (benchmark option 2)
- **Primary CSF platform (TBD):** Elecsys vs INNOTEST vs other.
- **Primary CSF metric (TBD):** Aβ42/40 ratio vs Aβ42 alone.
- **Primary CSF positivity rule (TBD):** explicit cutpoint + source/crosswalk.
- **Alternates:** platform-specific alternates or batch-adjusted definitions if needed.

### Plasma (index test)
- **Primary plasma analyte/assay (TBD):** specify biomarker and platform (e.g., p-tau217, p-tau181, Aβ42/40, etc.).
- **Primary plasma classification (recommended):** tri-category `Negative / Indeterminate / Positive`.
  - If ADNI provides only a continuous value, define two cutpoints to create an indeterminate range (TBD; prespecified method).
- **Sensitivity policies (binary mappings):**
  - Determinates-only: exclude indeterminate plasma results from 2×2.
  - Indet→Negative
  - Indet→Positive

## Clinical stage at index

- **Stage definition (TBD source):** CU / MCI / Dementia at the selected triad date.
- **Rule:** choose diagnosis closest in time to the triad median date; tie-breaker earliest.
- **Sensitivity:** stage at plasma date vs stage at PET date.

## Outcomes and reporting (Paper A)

### Paired benchmark dependence

Using the same triad cohort, compute plasma agreement against:
- PET benchmark (reference = PET positivity)
- CSF benchmark (reference = CSF positivity)

Report (per benchmark, overall + stage strata):
- `PPA = TP / (TP + FN)` and `NPA = TN / (TN + FP)`
- Paired comparator-swap drift: `ΔPPA = PPA_PET - PPA_CSF`, `ΔNPA = NPA_PET - NPA_CSF`
- Uncertainty: paired bootstrap across participants (TBD replicates; default 5000), reporting percentile 95% CIs.

### Gray-zone workflow and reflex confirmatory testing

Primary workflow:
- “Blood-first → confirmatory PET/LP only if plasma is Positive or Indeterminate.”

Core quantities (overall + stage strata):
- Confirmatory test rate: `Pr(plasma ∈ {Positive, Indeterminate})`
- Miss rate under a chosen benchmark: `Pr(plasma = Negative | benchmark = Positive)`
- Trade-off curves under sensitivity policies and (optionally) prevalence reweighting (TBD).

## Reviewer-proof supplement (minimum)

- Cohort flow + missingness summary
- Time-gap distributions for selected triads + sensitivity windows
- PET threshold sensitivity; CSF cutpoint/platform sensitivity
- Stage stratification and tracer/platform stratification
