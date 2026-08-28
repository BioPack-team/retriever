# Handoff: Generalizing Set Interpretation

This is a design/handoff document for a future effort to generalize and correctly
wire Retriever's TRAPI **set interpretation** feature. It is intentionally **not**
a prescriptive implementation plan — it captures the confirmed facts, the code
map, the defects, and the open decisions so a future branch can plan its own
revisions. All line numbers reflect the `tom` branch at the time of writing.

---

## Context — why this exists

"Set interpretation" is the TRAPI mechanism for asking about *sets* of entities
(`QNode.set_interpretation` ∈ `BATCH | ALL | MANY`). Retriever has a full
post-processing implementation of it (`evaluate_set_interpretation` in
`src/retriever/utils/trapi.py`), but it has **never actually run in production**:

- On `main`, the call site discards the return value — the function builds a new
  list and never mutates in place, so it is a silent no-op.
- On the `tom` branch (the TOM-migration work), the call was flipped on
  (`results = evaluate_set_interpretation(...)`, `qgx.py:232`). That surfaced the
  fact that the feature is **broken across two layers**, not merely un-wired.

The goal for the future session: **generalize and correctly wire** set
interpretation so ALL/MANY queries actually work end-to-end (today only the
`BATCH` no-op path is real).

---

## Ground truth — TRAPI spec (from `translator_tom.v1_6` QNode docstrings)

- **`set_interpretation`**: `BATCH` = each CURIE queried independently;
  **`ALL`** = *all* specified CURIEs MUST appear in each Result;
  **`MANY`** = members MUST form one or more sets in the Results (larger sets
  more desirable); missing/null ⇒ `BATCH`.
- **`ids`**: under `BATCH` holds the queried CURIEs; under `ALL`/`MANY` holds a
  **single UUID representing the set** (created via nodenorm, reused downstream
  for merge/cache).
- **`member_ids`**: the member CURIEs of the set; MUST be populated under
  `ALL`/`MANY`, MUST NOT be used under `BATCH`.
- There is **no `member_of` edge** concept anywhere in the model. (An earlier
  assumption that a `member_of` KG representation was required was investigated
  and **rejected** — TOM has no such helper and the repo has no usage. The set is
  represented solely by binding the set qnode to the UUID; the real member edges
  live in the collapsed result's `analyses[].edge_bindings`.)

---

## The core finding — the feature is broken at TWO layers

### Layer 0 — INPUT / expansion (the deepest problem, engine-side)

**Nothing in the lookup engine reads `member_ids` or `set_interpretation`.**
Grep of `src/` shows those fields appear only in `qgx.py` (import + the one call)
and in `trapi.py`. The engine drives everything off a QNode's `ids`:

- `Branch.get_start_branches` seeds branches from `node.ids_list`
  (`src/retriever/lookup/branch.py:316-332`).
- `SubqueryDispatcher.make_payloads` sets the *input* node's `ids` to the branch
  curie and leaves the *other* node's `ids` as-is from the qgraph
  (`src/retriever/lookup/subquery.py:240-254`).
- The operation planner (`src/retriever/metadata/optable.py`) plans on
  categories/predicates only; it never reads `ids`/`member_ids`.

Consequence: for a set QNode with `ids=[<uuid>]`, the engine queries the **UUID
literally**, the data tiers return nothing, and **no member-level results are
ever produced**. `evaluate_set_interpretation` is written to *collapse* results
that already bind the set qnode to real member CURIEs — but the real pipeline
never generates them. So even with Layer 1 fixed, ALL/MANY produce nothing.

### Layer 1 — POST-PROCESSING / collapse (`trapi.py`)

Confirmed defects in the collapse implementation:

- **B. Ordering.** `KnowledgeGraphDictUtil.prune` runs at `qgx.py:225`, *before*
  the collapse at `qgx.py:232`. The set UUID node is seeded into the KG at init
  (`initialize_kgraph`, `qgx.py:116` → `trapi.py:30-51`), but pre-collapse
  results never reference it, so prune deletes it — then collapse emits a result
  binding to the now-deleted UUID ⇒ **dangling KG reference**. Fix direction:
  run collapse before prune (then the UUID node survives because the collapsed
  result references it).
- **C. Two-node / single-edge assumption.** `_build_identifier_lookup_tables`
  reads only `node_entries[0]` and `node_entries[1]` per result
  (`trapi.py:557-561`). Any result with 3+ bound nodes (multi-hop) is silently
  mis-analyzed — the extra bindings are dropped from the adjacency table. All 8
  test fixtures are exactly 2-node/1-edge, so this is never exercised.
- **D. Hardcoded `QEdgeID("e0")`.** `_build_collapsed_result_analysis` keys the
  collapsed analysis `edge_bindings` under a literal `"e0"` (`trapi.py:654`),
  regardless of the real query edge id. Every fixture query graph uses edge id
  **`e01`**, yet the mixed fixtures' hand-authored results use `"e0"` to match
  the hardcode — so the tests pass while the production path would emit an
  edge-binding keyed to a qedge that doesn't exist in the query graph.
- **E. Minor (cosmetic).** In `_evaluate_node_connectivity`, the
  `len(subject_set) > 0` branch computes `missing_identifier_mapping` from
  `object_set.difference(...)` (`trapi.py:512-514`) where `object_set` is empty;
  it should use `subject_set`. Feeds only a debug log — no output impact.

---

## Test situation (important — the tests hide the bugs)

- **No integration test drives `qgx.execute()`** — the single production call
  site (`qgx.py:232`) is uncovered. No references to `qgx`/`QueryGraphExecutor`
  in `tests/`.
- All collapse logic is covered **only** by `tests/set_interpretation/`, whose
  `conftest.py` **hand-authors** `prefilter_results`/`postfilter_results`. Those
  fixtures bake in *both* broken assumptions: they are all 2-node/1-edge (C) and
  they key result edges on `"e0"` while the qgraph edge is `"e01"` (D). The tests
  assert on result **counts + per-result equality matched by `n0`/`n1` id**, not
  on realistic engine output.
- `initialize_kgraph` is separately (and validly) covered by
  `tests/test_utils_trapi.py::test_initialize_kgraph`.
- **Implication:** generalization needs new tests derived from *real* engine
  output (real qedge ids, real member-level bindings), plus a qgx→prune→response
  integration test that would have caught B and D. The existing fixtures should
  be re-authored, not trusted.

---

## Code map (call graph + key lines)

| Location | Role |
|---|---|
| `src/retriever/lookup/qgx.py:116` | `initialize_kgraph(self.qgraph)` — seeds KG with qnode `ids` (incl. set UUID) |
| `src/retriever/lookup/qgx.py:199-203` | results built via `part.as_result(...)` |
| `src/retriever/lookup/qgx.py:225` | `KnowledgeGraphDictUtil.prune(...)` — runs **before** collapse |
| `src/retriever/lookup/qgx.py:232-234` | **only** production call; note it re-serializes the model: `QueryGraphDict(**self.qgraph.to_dict())` |
| `src/retriever/lookup/branch.py:316-332` | start branches iterate `node.ids_list` (queries the UUID for set nodes) |
| `src/retriever/lookup/subquery.py:240-254` | data-tier payload uses `input_curie`; other node keeps qgraph `ids` |
| `src/retriever/lookup/partial.py:98-117` | `Partial.as_result` builds `node_bindings` from `(qnode_id, curie)` tuples |
| `src/retriever/metadata/optable.py` | planner: category/predicate only; no `ids`/`member_ids` |
| `src/retriever/utils/trapi.py:30-51` | `initialize_kgraph` |
| `src/retriever/utils/trapi.py:54-200` | `evaluate_set_interpretation` (entry; big semantics docstring) |
| `src/retriever/utils/trapi.py:203-246` | `_aggregate_node_groupings` (reads `member_ids` for ALL/MANY) |
| `src/retriever/utils/trapi.py:249-365` | `_evaluate_set_interpretation_all` (collapses + **prunes** partials) |
| `src/retriever/utils/trapi.py:368-433` | `_evaluate_set_interpretation_many` (collapses, **no** prune) |
| `src/retriever/utils/trapi.py:436-537` | `_evaluate_node_connectivity` |
| `src/retriever/utils/trapi.py:540-568` | `_build_identifier_lookup_tables` (**2-node assumption**, L557-561) |
| `src/retriever/utils/trapi.py:571-623` | `_build_collapsed_result_node_bindings` (validates `ids[0]` is UUIDv4, L610) |
| `src/retriever/utils/trapi.py:626-666` | `_build_collapsed_result_analysis` (**hardcoded `e0`**, L654) |
| `tests/set_interpretation/conftest.py` | 8 `mock_*_query` fixtures — all 2-node/1-edge `e01`; mixed fixtures key results on `e0` |
| `tests/set_interpretation/test_set_interpretation.py` | count + matched-equality assertions on the function in isolation |
| `tests/test_utils_trapi.py` | `test_initialize_kgraph` (only non-set-suite coverage in `trapi.py`) |

---

## Open decisions for the future session (design, not prescribed here)

1. **Product gate.** Should set interpretation be live at all? It has been a
   dormant no-op for its whole history; nothing depends on its output today.
   (The `tom` branch currently has it *activated but broken* at `qgx.py:232`;
   `main` discards the return. Whichever merges first sets the inherited state —
   decide keep-activated vs. revert-to-no-op as part of this work.)
2. **Where `member_ids` expansion belongs (the big fork).** Options to weigh:
   pre-processing the query graph before lookup (rewrite the set qnode to query
   its `member_ids`, then collapse after) vs. teaching the engine
   (`branch.py`/`subquery.py`) to expand `member_ids` natively vs. some hybrid.
   Without one of these, Layer 1 has no input.
3. **Generalizing shape.** Replace the `node_entries[0]/[1]` adjacency with a
   representation that handles multi-hop, multiple set nodes, set nodes on either
   edge end, and directionality — not just single-edge/2-node.
4. **Collapsed-edge identity.** Derive the collapsed analysis edge key from the
   actual qedge id(s) instead of literal `"e0"`; decide the representation when a
   collapse spans multiple qedges.
5. **Ordering in `qgx.execute`.** Where set interpretation runs relative to
   subclass solving (`solve_subclass_edges`, `qgx.py:214-221`) and `prune`
   (`qgx.py:225`).
6. **Model vs dict.** `trapi.py` set code is entirely dict-based and the call site
   re-serializes the `QueryGraph` model to a dict. Consider moving it onto TOM
   models for consistency with the completed migration (inward objects = models).
7. **MANY completeness.** Spec says "one or more sets"; the current code only
   forms the single maximal (all-members) set. Decide whether sub-set formation
   is required.
8. **KG/set-node richness.** The seeded UUID node is skeletal
   (`categories=[]`, `attributes=[]`). Decide whether it needs categories from the
   qnode or any set/aux-graph representation (no `member_of` — see spec above).

---

## Verification the future session should build toward

- Add an **integration test** exercising `QueryGraphExecutor.execute()` with a
  real ALL and MANY query (mock the data-tier driver to return member-level
  edges), asserting: (a) every collapsed result `node_binding` and analysis
  `edge_binding` resolves to a surviving KG node/edge (no dangling refs), and
  (b) ALL prunes partials while MANY keeps them.
- **Re-author** `tests/set_interpretation/conftest.py` from realistic engine
  output — real qedge ids, real member-level bindings — so the fixtures stop
  encoding defects C and D.
- Gate as usual: `ruff check src`, `basedpyright src`, `pytest -m "not live"`
  (set-interp suite + the new integration test).

---

## Session provenance (so decisions aren't relitigated)

- `member_of` KG representation: **investigated and rejected** (not in TOM/TRAPI).
- Confirmed the single production call site and single-file implementation — the
  post-processing layer is self-contained and safe to refactor in isolation.
- The Layer-0 finding (engine never expands `member_ids`) was verified by tracing
  `branch.py` / `subquery.py` / `optable.py` — it is the load-bearing discovery
  that turns "fix the collapse function" into "wire the feature end-to-end."
