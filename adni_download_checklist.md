# ADNI download checklist (Paper A triad table)

Goal: build a *triad cohort* where each participant has plasma + amyloid PET + CSF with usable dates and values.

## Minimum required tables (CSV)

1. **Clinical/demographics backbone**
   - Recommended: `ADNIMERGE` export (visit-level rows).
   - Must include: `RID`, visit identifier (`VISCODE`/`VISCODE2`), exam date, diagnosis/stage, age, sex (and APOE if present).

2. **Amyloid PET**
   - A table with scan date + tracer + a quantitative amyloid metric (SUVR or Centiloid) and/or a pre-derived positivity flag.
   - Must include: `RID`, scan date, tracer, metric value (and any region used).

3. **CSF amyloid**
   - A table with collection date + amyloid metric (Aβ42/40 ratio or Aβ42) and assay/platform information.
   - Must include: `RID`, collection date, metric value, platform (or enough metadata to infer platform/cutpoint).

4. **Plasma biomarker**
   - A table with draw date + plasma biomarker value(s) (e.g., p-tau217, p-tau181, Aβ42/40, GFAP, NfL).
   - Must include: `RID`, draw date, biomarker value(s) for the intended index test.

## Helpful add-ons (if not in ADNIMERGE)

- APOE genotype: `APOERES` export.
- Diagnosis summary: `DXSUM` / `PTDEMOG` / other ADNI clinical tables (only if ADNIMERGE is missing needed fields).

## What to share in the workspace

- Put the downloaded CSVs into `data/raw/` and tell me:
  - which plasma biomarker(s) you want as the Paper A “index test”
  - which PET metric to use (Centiloid vs SUVR) and desired positivity threshold(s)
  - which CSF platform/metric and cutpoint(s)

