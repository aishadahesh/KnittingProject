# Project rules

Read `README.md` for setup and navigation.
Read `next.md` if present, and verify its claims against the code.
Never read `todo.md`.

## Pivoting

- **Rapid Prototyping:** Optimize for quick experiments, not production.
  Preserve robot motion guards, explicit connection controls, calibration, and recorded experiments.
- **No Backward Compatibility:** Do not add compatibility layers or preserve obsolete interfaces unless requested.
  Breaking changes are acceptable when they simplify the requested solution.
- **Holistic Refactoring:** Update all affected callers and data structures when needed.

## Rule: Old code will live forever in the git history

- **YAGNI:** Delete confirmed unused code within the requested work and its dependencies; Git preserves committed code.
  Preserve unrelated and uncommitted user work.

## Rule: Less Code -> Fewer Bugs

- **Reduce to the Bone:** Prefer small, readable code.
  Remove duplication before adding abstractions.
- **Size Metric:** Reduce lines and characters through better data structures, not compressed formatting, removed whitespace, or unrelated statements on one line.
- **Programmatic Over Manual / Avoid String Mapping:** Replace repetitive field mappings with iteration over a shared schema.
  The schema must reduce total complexity; do not introduce reflection or indirection merely to reduce line count.
- **No Hidden Side Effects:** Keep transformations, lookups, and side effects explicit and separate from field mapping.

## Rule: The One True Path (TOTP)

- **Fixed Pipelines:** Route equivalent operations through one shared implementation.
  Keep orchestration linear where possible; branch when the algorithm requires it.
- **Signature Normalization:** Give interchangeable pipeline stages the same signature and return type.
  An unused parameter is acceptable when it preserves that interface.

## Rule: Single Source of Truth (SSOT)

- **One Implementation / DRY:** Maintain one implementation of each operation; do not copy-paste logic.
- **One Config / No Hardcoding:** Separate configuration from execution, with settings in one schema instead of scattered literals.
- **Atomic Updates / Valid State Guarantee:** Update source values and their dependencies as one operation.
  Restore all invariants before returning control or exposing state to another operation.
  Reads and rendering must never repair state.
- **Read-Only Purity:** Read and render functions must consume valid state without hidden mutations.

## Rule: The code is the documentation

- **No Comments / Code Documents Data:** Do not add code comments; express behavior and data contracts in code.
  Keep concise docstrings for units, coordinate frames, mathematical assumptions, and resource ownership that names and types cannot express.
  Preserve existing explanations of hardware safety and numerical invariants.
- **Avoid Prose:** Keep logs, errors, and explanations short, specific, and informative.
- **Code Is Source of Truth:** Code establishes current behavior; these rules establish how to change it.
  Fix or delete inaccurate documentation in the same commit as the code change.

## Structural rules

- **Purposeful Abstraction:** Extract functions for reuse, architectural boundaries, or complex logic that cannot reasonably be simplified.
  Do not extract merely to name a code block.
- **Single-Use Functions:** Keep single-use helpers local, or inline them when short.
  Entry points, callbacks, and module interfaces are architectural boundaries.
- **Functional by Default:** Prefer pure functions, dictionaries, and closures.
  Add a class only when state requires encapsulation and the user explicitly approves it.
  Existing state, renderer, camera, transport, and worker classes own resources and lifecycles; do not replace them solely to avoid classes.
- **Flat Architecture:** Use guard clauses and early returns.
- **Execution-Order Layout:** Arrange code in execution order.
- **Locality:** Define variables and helpers near their use; centralize utilities reused across modules.
- **Scope-Aware Naming:** Use concise local names and descriptive shared names.
- **Appearance Is Structure:** Align repetitive code where it makes differences easier to scan.

## Fail-Fast Core & Boundary Error Handling

- **Push Try/Catch to the Edge:** Let core logic fail fast and propagate exceptions.
- **Boundary Safety:** Catch failures at application boundaries such as UI callbacks and file I/O.
  Report useful context and recover only when valid state can be preserved.
  Never suppress failures silently.

## Laser-Focused Scope Compliance

- **Strict Adherence:** Implement the requested behavior without speculative features or wrappers.
- **Scope Is Features, Not Diff Size:** Change related files and signatures when required for a direct solution.
  Do not add workarounds merely to avoid those edits.
  Limit cleanup to the requested work and its dependencies.
- **Critical Agent Protocol:** Before editing code, check that the planned change follows these rules.
  Do not use existing violations as a reason to introduce more.

## Verification

- **Verify Changed Behavior:** Run checks relevant to the changed behavior.
  Add regression tests for substantive bug fixes.
  Report what was verified and any remaining uncertainty.
- Run `uv run ruff check` and `uv run pytest` with the camera extra installed.
  Use temporary directories, dry transports, and offscreen rendering for automated checks.
  Never connect to or move a real robot as part of a cleanup check.

## Project architecture

- **Stateless Core:** Keep geometry and fitting functions independent of GUI state, file access, and runtime setup.
- **Pure Spline Geometry Pipeline:** Use `knitting_core.build_spline_mesh` for knitted meshes and the shared centreline evaluator for simulation.
  Crochet has its own stitch geometry; do not force these different algorithms into one implementation.
- **Centralized State:** Keep editable application values in `AppState._data`, defaults and saved field declarations in `config/state_schema.json`, and live resources with their owning controllers.
- **One Dependency Manifest:** Declare application dependencies and extras in `pyproject.toml`; resolve them with `uv.lock`.
  Splat training has its own environment managed by `scripts/setup_splatting.ps1`.
- **Project Paths:** Declare application paths in `src/paths.py`.
  CLI arguments and entry-point imports may resolve their own supplied paths.

## Git rules

- **Review Existing Work:** Review existing changes before editing.
  Preserve pre-existing work and keep it out of task commits unless the user asks to include it.
- **Work in a Branch:** Work on a feature branch, not `main` or a detached HEAD.
  If work started on `main`, move its commits to a feature branch.
- **Commit Before Handoff:** Commit task changes before handoff; do not ask whether to commit.
- **Small-Work Exception:** Before final handoff, integrate completed small work that can safely be squashed to one commit and rebased cleanly.
  Squash it, rebase onto `main`, fast-forward `main`, and delete the temporary branch.
  Do not preserve a dangling or archived branch for this case.
  If integration cannot be completed safely, preserve the branch and explain the blocker.
- **Integration Boundary:** Automatic integration applies only when it brings in this task's commits alone and preserves uncommitted work.
  If the starting feature branch contains unrelated commits absent from `main`, keep the cleanup branch for review instead of merging that feature history.
  Do not push or change remote branches without a request.
- **Squash-Merge:** For other completed features, duplicate the branch, squash the copy into a few meaningful commits, and merge the copy into `main`.
  Preserve the original feature history as `merged-<original-name>` and record the resulting commit hash in that branch's description.

## Documentation

| File | Purpose |
| --- | --- |
| `AGENTS.md` | Project rules |
| `README.md` | Setup, navigation, and non-obvious constraints; at most 400 lines |
| `next.md` | Previous session results and next actions |
| `todo.md` | Optional side tasks; prepend entries, never read |

- Create Markdown files only when requested, except `next.md` as specified below.
- Markdown has no column limit; do not hard-wrap prose to a fixed width.
  Start every sentence on a new source line and keep each sentence on that line.
- Preserve explicit rule names and acronyms so the user can refer to them directly, such as TOTP and SSOT.
- Document intent, constraints, and non-obvious decisions.
  Keep navigation brief; do not duplicate implementations or data schemas in prose.
- Name files and symbols, not line numbers.
- Keep project history in Git.
  Use `next.md` only for session continuity: create or update it before handoff when unfinished work needs a later session.
  Record the current state, blockers, and concrete next actions; replace stale entries.
  Completed work needs no new handoff file.
- Keep the README within its cap by shortening existing content when adding more.
