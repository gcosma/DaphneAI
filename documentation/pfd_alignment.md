## PFD (Regulation 28) – Recommendation ↔ Response Alignment

This note documents how we approach **Prevention of Future Deaths (PFD)** style documents
(Regulation 28) in v2, and how that differs from “explicit recommendation reports”.

The intent is to keep these behaviours **profile-scoped** (so they don’t leak into the
`explicit_recs` pipeline) and to make alignment semantics explicit (so we don’t “force”
matches that are jurisdictionally impossible).

### Why PFD is different

PFD ecosystems frequently include:
- A long-form report (sometimes narrative/judgment-like) that contains concerns and embedded
  directives addressed to multiple bodies (Home Office, Police, Justice, NHS Trusts, etc.).
- Multiple response letters (one per addressee) that respond only to the subset relevant
  to that organization, and sometimes also to general findings.

This means alignment is often **many directives → few response documents**, with many
directives correctly having **no in-scope match** in a given response.

Many PFD reports are also **“null explicit recommendation”** cases: they contain no clean
directive sentences (“I recommend…”, “X should…”), but still enumerate *Matters of Concern*
that recipients must respond to. We treat these concerns as a separate extraction target
instead of relabeling them as “recommendations”.

### Terminology (v2)

v2 uses `Recommendation.rec_type` to make the extraction channel explicit:
- `numbered`: explicit “Recommendation N …” headings (profile: `explicit_recs`)
- `action_verb`: action-verb sentences outside primary spans (shared channel; semantics aligned to v1)
- `pfd_concern`: response-required concerns (often numbered items under “MATTERS OF CONCERN” / “CORONER’S CONCERNS”)
- `pfd_directive`: embedded directive sentences in long-form narrative PFD reports (profile: `pfd_report`, currently opt-in)

Semantic targets (useful for evaluation and UI framing):
- **Target A (explicit recommendations):** directive sentences when present (often absent).
- **Target B (response-required concerns):** the concerns/failures the recipient is expected to address in their response.

Heuristic inventory for Target B extraction (windowing, boilerplate exclusions, common concern patterns)
is consolidated in `documentation/pfd_rule_inventory.md`.

For PFD response documents we expect to introduce response “blocks”:
- A **response block** is a thematic section (e.g., “Prevent”, “MAPPA”, “Operation Plato”) rather
  than a single sentence.

### Recommendation extraction – PFD profile (`pfd_report`)

PFD recommendations are extracted in two structural ways:

1) **Short-form concerns list** (`pfd_concern`)
- Anchor: “MATTERS OF CONCERN” / “CORONER’S CONCERNS”
- Unit: each numbered item like `(1)`, `(2)`, `(3)` (and narrative concern sentences within the concerns window)
- End: typically “ACTION SHOULD BE TAKEN” / “YOUR RESPONSE”
- When this structure exists, it is usually the cleanest “unit that expects a response”.

2) **Long-form narrative directives** (`pfd_directive`)
- Some documents use paragraph numbering throughout the report and do not contain a short,
  clean concerns list. In these cases, numbering is not “concern numbering” and extracting
  “all numbered points” creates unusable noise.
- Unit: directive sentences such as:
  - “I recommend / I strongly recommend …”
  - “I request / I suggest / I encourage …”
  - “It is vital/critical/important that …”
  - “<addressee> should <action verb> …”

**Heuristic refinement guidance (report8-style)**
- Keep: forward-looking, actionable obligations (monitor/evaluate/ensure/provide reassurance/assess/put in place).
- Drop: statutory boilerplate (“action should be taken…”, “duty of those receiving this report…”), reported
  speech (“Mr X agreed that … should …”), and narrative conclusion/citation blocks.

**Practical note (current priorities)**
- The “action-verb” channel is treated as the canonical continuity method and is aligned to v1 semantics.
- Longer-term, we expect to collapse multiple PFD “recommendation-like” channels into a single, explainable
  action-verb-oriented surface in the UI, while still preserving the semantic distinction between explicit
  recommendations (Target A) and response-required concerns (Target B) for downstream alignment and evaluation.

### Response extraction – PFD response documents (proposed “block” approach)

PFD responses often use **thematic headings** rather than “Government response to recommendation N”.

Recommended unit: **thematic blocks** delimited by headings.

Robust heading segmentation ladder:
1) **Intro-list look-back**
   - Many PFD responses list the headings they will use in an introduction.
   - Treat these as “expected headers” and split later merged lines at those phrases.
2) **Title-case + sentence-case collision**
   - Detect lines like “Prevent I acknowledge …” and split between “Prevent” and “I acknowledge”.
3) **Strong keyword priority**
   - Hard breaks for domain keywords that commonly function as headings in this genre:
     `MAPPA`, `Prevent`, `Pathfinder`, `Operation <Name>`, etc.

### Alignment semantics – “scoped matching”

PFD alignment is not “best response per recommendation”. It is:
- **in-scope match** when the directive is addressed to the responder (or a joint responder),
- **out-of-scope** when it is addressed to a different agency, and
- **response-to-findings** when a response block is legitimate but not tied to a single directive sentence.

For Target B (response-required concerns), alignment is conceptually “which response block addresses this concern?”,
often without a single-sentence directive anchor. In practice this will likely rely on:
- concerns-window extraction + deduplication
- topic overlap between concern text and response blocks
- optional responder-scope gating when addressees can be inferred

#### Step 1: Addressee extraction (gatekeeper)

Use a simple hierarchy of text patterns to tag addressees:
1) **Agency-subject pattern (active voice)**
   - `<Entity> (should|must|is to|to) <action verb> …`
2) **Prepositional target (passive / request)**
   - `I (recommend|request|suggest|encourage) … (by|to) <Entity> …`
3) **Compound coordination**
   - `… encourage <A> and <B> to work together …` → addressees = {A, B}

Normalization: map extracted surface forms to a small canonical taxonomy (e.g. `HOME_OFFICE`,
`POLICE`, `JUSTICE`, `HEALTH_TRUST`). Track multiple addressees when present.

#### Step 2: Scope gating

If the directive addressee set does not include the responder, mark it:
- `out_of_scope` (correct outcome; do not force a match)

If it includes multiple addressees and only one has responded so far:
- `partial` (the response is valid but not complete across addressees)

#### Step 3: Topic mapping (directive → response block)

For directives that are in-scope:
- Match by overlap between directive keywords and response block headings/topics
  (e.g. `MAPPA`, `Prevent`, `Operation Plato`, `intelligence`).
- Then choose an excerpt/snippet inside the block that describes action taken/proposed.

#### Step 4: “valid response” vs deferral

Some passages are not actions; they are deferrals or jurisdiction statements.
Treat these specially:
- `deferral`: “responsibility of X”, “falls under X”, “I anticipate X will respond…”

### Error checklist (what counts as a hallucination)

An alignment is an error if it violates jurisdiction or intent:
- **Jurisdiction mismatch:** directive explicitly for Agency A, response is from Agency B.
- **Deferral treated as action:** “not our responsibility” counted as “response match”.
- **Temporal mismatch:** future directive matched to description of the criticized old system
  rather than a new measure (“we have since introduced…”, “we will…”, “a review has been commissioned…”).
- **Keyword trap:** shared noun only (e.g. “risk”) but mismatched modifiers/topics (extremism vs self-harm).

### UI guidance (human-facing)

- Show directive sentence as the “recommendation”, but provide **context expansion** (prior paragraphs).
- Clearly label `out_of_scope` directives (so “no match” doesn’t look like a system failure).
- Allow unmatched response blocks labeled as **“General response / response to findings”**.

### Validation workflow (human-in-the-loop)

For a new PFD pair:
1) Preprocessing inspection:
   - `python -m tools.preproc_v2_preview --pdf <report>.pdf`
   - `python -m tools.preproc_v2_preview --pdf <response>.pdf`
2) Recommendation inspection (PFD profile):
   - `python -m tools.inspect_recs_v2 --pdf <report>.pdf --profile pfd_report`
   - Optional meeting view: `streamlit run pfd_compare_app.py` (Action Verbs vs Full concerns vs Extended Action Verbs, all run over the same v2-preprocessed text baseline).
3) Response inspection:
   - Use `python -m tools.inspect_pfd_alignment --report <report>.pdf --response <response>.pdf`
     to preview inferred responder identity, response blocks, scoped matches, and unmatched
     blocks (candidate: response-to-findings).
