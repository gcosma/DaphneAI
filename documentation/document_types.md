## Document Types

What inputs the system targets and what outputs we expect.

### Coroners’ reports
- Nature: narrative reports with recommendations and responses embedded in prose.
- Typical input: PDF converted to text; pages stored with page numbers; recommendations often prefixed or enumerated.
- Expected output: extracted recommendations (text + ids/labels), aligned responses/snippets with status and similarity.
- Gotchas: OCR noise, variable numbering schemes, long sentences that need careful context windows.

### Other reports (e.g., “Matt’s file” set)
- Nature: structured or semi-structured institutional reports with action items and follow-ups.
- Typical input: similar PDF/text ingestion; recommendations may be in bullet lists or tables.
- Expected output: same schema as above; table-derived items may need preprocessing to retain structure.

### Guidance for new document types
- Capture sample fragments (input) and desired parsed outputs early; add to tests/fixtures when available.
- Note any domain-specific markers (ids, verb phrases) that affect extraction or alignment; update patterns accordingly.
