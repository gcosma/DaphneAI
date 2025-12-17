## Document Types

What inputs the system targets and what outputs we expect.

### Coroners’ reports (PFD / Regulation 28)
- Nature: narrative reports that raise concerns about systemic failures and require recipients to respond. Many contain **no explicit “recommendations”** in the “X should do Y” sense.
- Typical input: PDF; v2 prefers layout-aware extraction (`unstructured` + `syntok`), and for inspection we often work from sentence-per-line dumps.
- Expected output (v2 mental model):
  - **Explicit recommendations (Target A)** when present: directive sentences (“I recommend…”, “It is vital that…”, “<Entity> should…”).
  - **Response-required concerns (Target B)**: enumerated “MATTERS OF CONCERN” / “CORONER’S CONCERNS” items and other concrete deficiencies that the recipient must address in their response.
  - Downstream: responses segmented and aligned to the relevant concerns/directives, with “out of scope” semantics when responder identity does not match addressee.
- Gotchas:
  - Statutory boilerplate (“In my opinion action should be taken…”, duty-to-respond deadlines) contains misleading “should” language and must be excluded.
  - Concerns are often written in past tense (“There was no…”, “No action was taken…”) and are still response-relevant.
  - Some documents contain multiple responders/letters; alignment must be scoped to responder identity.

### Other reports (e.g., “Matt’s file” set)
- Nature: structured or semi-structured institutional reports with action items and follow-ups.
- Typical input: similar PDF/text ingestion; recommendations may be in bullet lists or tables.
- Expected output: same schema as above; table-derived items may need preprocessing to retain structure.

### Guidance for new document types
- Capture sample fragments (input) and desired parsed outputs early; add to tests/fixtures when available.
- Note any domain-specific markers (ids, verb phrases) that affect extraction or alignment; update patterns accordingly.
