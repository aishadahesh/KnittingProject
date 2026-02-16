# Project Workflow

## Guiding Principles

1. **The Plan is the Source of Truth:** All work must be tracked in `plan.md`
2. **The Tech Stack is Deliberate:** Changes to the tech stack must be documented in `tech-stack.md` *before* implementation
3. **Research-Driven Development:** Focus on mathematical correctness and experimental reproducibility.
4. **User Experience First:** Every decision should prioritize user experience and research utility.
5. **Non-Interactive & CI-Aware:** Prefer non-interactive commands. Use `CI=true` for watch-mode tools (tests, linters) to ensure single execution.

## Task Workflow

All tasks follow a strict lifecycle:

### Standard Task Workflow

1. **Select Task:** Choose the next available task from `plan.md` in sequential order

2. **Mark In Progress:** Before beginning work, edit `plan.md` and change the task from `[ ]` to `[~]`

3. **Implementation:**
   - Implement the feature or fix as described in the task.
   - Ensure the implementation aligns with the research goals and tech stack.

4. **Verification:**
   - Verify the changes manually or through relevant research scripts.
   - Ensure that the differentiable pipeline remains functional if geometry or rendering logic was modified.

5. **Update Plan:**
   - Once the task is complete, update its status in `plan.md` from `[~]` to `[x]`.

### Phase Completion Verification

**Trigger:** This protocol is executed immediately after a task is completed that also concludes a phase in `plan.md`.

1. **Announce Phase Completion:** Inform the user that the phase is complete.
2. **Manual Verification:** Propose a manual verification plan to the user to ensure the research goals of the phase have been met.
3. **Await Approval:** Wait for the user to confirm that the phase meets their expectations.

## Quality Gates

Before marking any task complete, verify:

- [ ] Implementation meets task description
- [ ] Research goals are addressed
- [ ] Code follows project's code style guidelines
- [ ] Differentiable rendering pipeline is intact (if applicable)
- [ ] Documentation updated if needed

## Development Commands

### Setup
```bash
# Install dependencies
pip install mitsuba drjit jax jaxlib optax torch torchvision matplotlib pillow vedo
```

### Running the Optimizer
```bash
python optimize_knitting.py
```

## Definition of Done

A task is complete when:

1. All code implemented to specification
2. Research goals for the task are met
3. Status updated in `plan.md`

## Manual Commits
**Note:** The AI agent will not perform automatic commits or attach git notes. All version control operations (add, commit, push) are to be handled manually by the user.
