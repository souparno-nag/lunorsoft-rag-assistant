# Project conventions

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
