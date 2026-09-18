# Project conventions

## Starting work

Do not execute a task unless I explicitly say so. Reading the codebase, reading
the specs in `specs/`, analysing what is there and proposing an approach are
always fine and need no permission; creating or editing project files does.

When a task or phase looks ready to start, summarise the current state, flag
whatever decisions are open, and then wait. An obvious next step in
`specs/tasks.md` is not an instruction to begin it — I decide the order and
shape of the work.

This does not narrow the Git rule below: once I have asked for a task, commit
its result without pausing for confirmation.

## Git

Commit after every meaningful change. In practice that means one commit per
completed task from `specs/tasks.md`, referencing the task ID in the subject
(e.g. `T0.2: add requirements.txt and pin Python 3.12`). Do not batch several
finished tasks into one commit, and do not leave completed work uncommitted.

Committing does not need to be confirmed each time. Pushing, force-pushing, and
opening pull requests still do.

Do not add AI attribution to commit messages or pull request descriptions — no
`Co-Authored-By` trailer for an assistant, no "Generated with" footer.

## Task tracking

`specs/tasks.md` has a Status column (☐ / ◐ / ☑). Flip a task to ☑ in the same
commit that completes it, so the spec never disagrees with the repository.
