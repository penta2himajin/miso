# Session Handoff Protocol

Protocol for handing off long-running work between agent sessions via files under `docs/handoff/` (ADR_002 D13).

## Premises

- **Scope**: long-running workstreams that span multiple sessions and possibly multiple interfaces.
- **Medium**: one Markdown file per workstream, `docs/handoff/<workstream>.md`, committed and pushed.
- **Granularity**: one file per workstream. Not one per session.
- **Avoid destructive history loss**: the sections above `## Session log` hold current state and are overwritten each session. `## Session log` at the bottom is append-only.
- **Template**: `docs/handoff/_template.md`.

## Sender procedure (end of session)

1. Overwrite the current-state sections of the workstream file.
   - **Snapshot**: branch + latest commit SHA + timestamp. Record the SHA of the last *work* commit, since the handoff commit itself comes after it.
   - **Next action**: granular enough that the next session can begin immediately.
   - **Failed approaches**: record anything tried and abandoned. Skipping this is the most common cause of duplicated effort.
2. Append this session's outcome to `## Session log` as a single dated paragraph.
3. Commit (`docs(handoff): …`) on the working `ai-written/<topic>` branch and push it (ADR_003 D19). The working tree must be clean afterwards, or the dirty state is recorded in the Snapshot.
4. Reference the handoff file path in the final message.

## Receiver procedure (start of session)

1. List `docs/handoff/*.md` (excluding `_template.md`) and identify the relevant workstream.
2. Read the file. Internalise Snapshot, Next action, Failed approaches.
3. Verify the Snapshot matches reality.
   - `git fetch` and compare `git log -1 origin/<branch>` with the recorded SHA (the handoff commit may follow it).
   - Check working tree state if claimed clean.
4. If drift exists, report to the user before acting and request guidance.
5. If no drift, read the **Next action** aloud to the user and confirm before executing.
6. Execute. Verify against the **Verification** field. Then follow the sender procedure.

## Promotion rule

When a Decision in a handoff file is referenced in 2+ later sessions, promote it to an ADR under `docs/decisions/`. Replace the entry in the handoff file with a link to the ADR.

## SSOT precedence

When sources conflict:

1. `CLAUDE.md` / `AGENTS.md` (project-wide invariants)
2. `.claude/rules/` (path-scoped)
3. ADR in `docs/decisions/` (settled judgements)
4. Handoff file (current workstream state)

## Anti-patterns

- Treating the current-state sections as a chat log. History goes in `## Session log`.
- Inlining large logs or code dumps. Use file path + SHA + line-range pointers instead.
- Ending a session with no concrete Next action.
- Leaving the handoff update uncommitted or unpushed.
- Deleting a finished workstream file without migrating its Decisions to an ADR.

## Parallel workstreams

Multiple workstream files are fine. Express dependencies by linking the other file.
