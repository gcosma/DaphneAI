# PFD Extraction – Rule Inventory (Draft)

This note consolidates common patterns observed across five independently produced analyses in
`output/pfds_v2_preproc/pfd_analysis_report/*.md`. The goal is to provide an implementation-ready
reminder of the heuristics we likely want to encode for PFD-style documents.

## Targets

PFD documents often contain **no explicit recommendation sentences** (“null A”). To keep semantics
clean and avoid conflating legal boilerplate with actionable content, we treat extraction as two
separate targets:

- **Target A: Explicit recommendations**
  - Sentences that explicitly direct future action (e.g., “I recommend…”, “X should…”, “It is vital
    that…”).
- **Target B: Response-required concerns**
  - Concrete failures/deficiencies/risks the recipient is expected to address in their response,
    often under “MATTERS OF CONCERN” / “CORONER’S CONCERNS”, typically written in descriptive or
    past-tense form.

In the five PFD examples reviewed, Target A was consistently absent, while Target B was present and
high-signal once localized to the concerns section.

## High-level pipeline (recommended ordering)

1) **Scope**: infer `CONCERNS_WINDOW` boundaries from headings.
2) **Exclude**: aggressively remove statutory/procedural boilerplate (global).
3) **Include B**: extract concern sentences inside `CONCERNS_WINDOW` using failure/concern patterns.
4) **Disambiguate A**: exclude “nested should” constructs from being treated as explicit directives.
5) **Deduplicate**: prefer concerns-window instances over narrative duplicates elsewhere.

## Rule inventory (candidate rules)

Notes:
- Regexes below are intended as single-line, case-insensitive patterns (PCRE-style).
- “Scope” rules describe window boundaries; downstream rules can reference these windows.
- “EXCLUDE_A” indicates “exclude from Target A classification”, not “discard entirely”.

### Scope rules

- **SCOPE_CONCERNS_START** (SCOPE)
  - Start `CONCERNS_WINDOW` on common headers:
  - Regex: `(?i)^(?:the\\s+)?(?:matters?\\s+of\\s+concern|coroner(?:'s|s)?\\s+concerns)\\b`

- **SCOPE_CONCERNS_END** (SCOPE)
  - End `CONCERNS_WINDOW` at the statutory transition header:
  - Regex: `(?i)^action\\s+should\\s+be\\s+taken\\b`
  - Optional additional end markers (some documents): `(?i)^your\\s+response\\b|^copies\\b|^copies\\s+and\\s+publication\\b`

- **SCOPE_BOILERPLATE_START** (SCOPE)
  - Start `BOILERPLATE_WINDOW` at:
  - Regex: `(?i)^action\\s+should\\s+be\\s+taken\\b`

### Boilerplate/procedure exclusions (global)

- **EXCLUDE_STATUTORY_TRIGGER** (EXCLUDE)
  - The canonical legal trigger phrase:
  - Regex: `(?i)^in\\s+my\\s+opinion\\s+action\\s+should\\s+be\\s+taken\\s+to\\s+prevent\\s+future\\s+deaths\\b`

- **EXCLUDE_STATUTORY_RISK_PREFACE** (EXCLUDE)
  - The canonical “risk unless action is taken” preface:
  - Regex: `(?i)^in\\s+my\\s+opinion\\s+there\\s+is\\s+a\\s+risk\\s+that\\s+future\\s+deaths?\\s+(?:could|will)\\s+occur\\s+unless\\s+action\\s+is\\s+taken\\b`

- **EXCLUDE_RESPONSE_DUTY_DEADLINE** (EXCLUDE)
  - Duty-to-respond and deadline instructions:
  - Regex: `(?i)\\bunder\\s+a\\s+duty\\s+to\\s+respond\\b|\\bwithin\\s+56\\s+days\\b|\\byour\\s+response\\s+must\\s+contain\\b`

- **EXCLUDE_ADMIN_COPIES_PUBLICATION** (EXCLUDE)
  - Distribution/publication sections (tune conservatively; some lines may contain “concerned”
    but function administratively):
  - Regex: `(?i)^copies\\b|^copies\\s+and\\s+publication\\b|\\bpublication\\b|\\breport\\s+is\\s+being\\s+sent\\b|\\bwill\\s+be\\s+copied\\b`

### Target B inclusions (within concerns window)

- **INCLUDE_B_ENUMERATED_ITEM** (INCLUDE_B)
  - Numbered/bulleted list items are high-signal inside `CONCERNS_WINDOW`:
  - Regex: `(?i)^(?:\\(\\d+\\)|\\d+\\.|[ivx]{1,6}\\)|[•\\-*])\\s+`

- **INCLUDE_B_FAILURE_TEMPLATES** (INCLUDE_B)
  - Deficit/failure templates seen repeatedly:
  - Regex (starter set):
    - `(?i)^there\\s+was\\s+(?:no|a\\s+lack\\s+of|a\\s+failure\\s+to)\\b`
    - `(?i)^no\\s+action\\s+was\\s+taken\\b`
    - `(?i)\\b(did\\s+not|failed\\s+to)\\b`

- **INCLUDE_B_FIRST_PERSON_CONCERN** (INCLUDE_B)
  - Narrative concerns expressed explicitly by the coroner:
  - Regex: `(?i)^i\\s+(?:am|remain|was)\\s+(?:concerned|alarmed|worried)\\b`

- **INCLUDE_B_RISK_SUMMARY_NONBOILERPLATE** (INCLUDE_B)
  - Risk summary phrasing that is *not* the statutory preface:
  - Regex: `(?i)\\bgives?\\s+rise\\s+to\\s+(?:a\\s+)?risk\\b`
  - Guardrails: only accept when scoped to `CONCERNS_WINDOW` and not matching the statutory
    “in my opinion…” patterns above.

- **INCLUDE_B_PERSISTENT_DEFICIT** (INCLUDE_B)
  - Persistent deficit phrasing (e.g., “there remains … no …”):
  - Regex: `(?i)^there\\s+remains\\b.*\\bno\\b`

- **INCLUDE_B_SYSTEM_GAP_NO_FEATURE** (INCLUDE_B)
  - “There are no …” system gaps (algorithms/features/requirements):
  - Regex: `(?i)\\bthere\\s+are\\s+no\\b.*\\b(?:feature|algorithm|requirement|process|procedure|policy|plan|training|staff|system)\\b`

### Target A disambiguation (global)

- **EXCLUDE_NESTED_SHOULD_AS_A** (EXCLUDE_A)
  - Prevent false “A” hits when “should” appears in embedded clauses describing past failures:
  - Regex: `(?i)\\b(?:whether|determine|consider)\\b.*\\bshould\\b`
  - Note: this does *not* exclude a sentence from Target B if it is otherwise a failure/concern.

### Post-processing

- **DEDUP_WITHIN_DOC** (POST)
  - When near-duplicate concern statements appear both in narrative (“circumstances”) and in the
    concerns list/window, prefer the concerns-window instance to avoid duplication.
  - Suggested logic: normalize whitespace + lowercase; compare prefix keys or token overlap; if a
    later concerns-window sentence substantially contains an earlier narrative sentence, drop the
    earlier one.

## Known ambiguity (flag for explicit policy)

Some PFDs contain phrasing like “action should be taken by the Trust to resolve this” embedded as a
trailing clause of a risk sentence. Across analyses, this was consistently treated as **not a clean
Target A explicit recommendation** (generic action, “resolve this”), but may still be extracted as
Target B if it is inside `CONCERNS_WINDOW` and tied to a specific preceding deficiency.

We should make this decision explicit in code/tests when we implement Target A for PFDs.

