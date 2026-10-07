# Instructions for AI coding agents

## Attribution

- Never add AI attribution of any kind: no `Co-Authored-By` trailers, no
  "Generated with ..." lines, no AI-tool mentions in commit messages, pull
  requests, issues, code comments or docs.
- Commits are authored by the repository owner only.

## Working on this repository

- Read `SPEC.md` before changing behaviour; the code must not contradict it.
- Keep it minimal and readable: one file per model, no forecasting frameworks.
- Respect the information set (SPEC §2): `tests/test_models.py::test_no_leakage`
  must keep passing.
- Before committing: `make lint && make test`.
