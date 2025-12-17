## Evaluation (placeholder)

Sketch of how to report and compare performance by document type.

- Datasets: list document types and sample sizes once defined (e.g., coroners’ reports, institutional reports).
- Metrics:
  - GOV.UK-style reports: numbered recommendation recall, structured response recall, and numbered alignment correctness.
  - PFD reports: treat extraction as two targets:
    - Target A (explicit recommendations): precision-first (often null).
    - Target B (response-required concerns): precision/coverage within the concerns window; boilerplate false-positive rate should be ~0.
  - Alignment: matched/in-scope/out-of-scope/deferral rates (PFD) and matched/partial/none (non-PFD).
  - Search: precision/recall per mode as needed.
- Procedure: describe ingestion, extraction profiles (v1 vs v2, explicit vs PFD), alignment steps, and how scores are computed.
- Results: summarize per document type; highlight failure modes and qualitative observations.
- Reproducibility: note configs, model versions, and thresholds used in experiments.
