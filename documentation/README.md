## Documentation plan

This folder will hold concise, task-focused docs for the DaphneAI codebase. The goal is to help new contributors quickly find the right entry points and to support future technical reporting without creating a monolithic guide.

### Proposed files
- `architecture.md` — high-level system layout: core vs UI packages, data flow through ingestion, search, extraction, alignment, and display; key entrypoints.
- `alignment.md` — alignment logic overview: recommendation extraction assumptions, response detection, alignment heuristics, where to extend/improve, and invariants to keep intact.
- `search.md` — search modes (smart/exact/fuzzy/semantic/hybrid), main helpers, and guidelines for adding new search strategies.
- `ui-guide.md` — Streamlit UI structure: tab/module mapping, how UI composes with core services, and tips for adding new UI features without leaking business logic.
- `deployment.md` — running locally vs. packaging, environment variables, models, and performance considerations (keep minimal, link to scripts if added later).

### Writing principles
- Keep each file <400–500 lines; prefer links between files over repetition.
- Lead with diagrams or short flows where helpful; keep prose crisp and actionable.
- Call out extension points and guardrails (what not to break) in each area.
