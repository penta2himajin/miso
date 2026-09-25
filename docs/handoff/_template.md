# Handoff: <workstream>

<!--
Current-state sections (everything above "Session log") are overwritten each session.
"Session log" is append-only. Protocol: docs/handoff-protocol.md.
Decisions referenced in 2+ sessions migrate to docs/decisions/ (ADR).
-->

## Snapshot

- Branch: `<branch-name>`
- Last work commit: `<short-sha>` @ YYYY-MM-DD
- Working tree: clean / dirty (`<file1>`, `<file2>`)
- Last session: YYYY-MM-DD HH:MM <TZ>

## Status

<!-- one of: in-progress / blocked / ready-for-review -->

## Next action

<!-- One sentence. Verb + object + expected outcome.
The receiving session reads this aloud and confirms with the user before executing. -->

## Verification

<!-- How to know the next action is complete. Command + expected output + fallback check. -->

## Context pointers

<!-- Pointers, not content. Use commit SHAs and line ranges where possible.
- Code: `path/to/file` L42-78 @ <sha>
- Doc: `docs/<...>.md` §<section> -->

## Decisions made

<!-- Settled within this workstream; do not re-litigate. -->

## Failed approaches

<!-- Mandatory if anything was tried and abandoned.
- Approach
- Why it failed (verbatim error message if any)
- Why the current approach is preferred -->

## Open questions for user

<!-- Items needing user input before proceeding. Delete section if empty. -->

## Session log

<!-- Append-only. One dated paragraph per session. -->
