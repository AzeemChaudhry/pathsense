# Security policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub private vulnerability reporting](https://github.com/AzeemChaudhry/pathsense/security/advisories/new),
not in a public issue. You will get a reply within a few days.

## How the repository is protected

| Area | Protection |
| --- | --- |
| `main` branch | Changes only through pull requests with passing CI (`lint`, `test (3.11)`, `test (3.12)`); signed commits required; no force pushes or deletion |
| Release tags `v*` | Only the maintainer can create, move or delete them; releases are published only from signed, verified tags on `main` whose commit passed CI |
| Pull requests from forks | Workflows run only after the maintainer approves them, never with repository secrets, and with a read only token |
| `@claude` reviews | Triggered only by the owner or collaborators; runs with a read only token and no file editing tools, so it can comment but not push; the action is pinned to a commit |
| GitHub Actions | Only GitHub's own actions plus `anthropics/claude-code-action` and `astral-sh/setup-uv` may run; the default token is read only |
| Secrets | Stored as encrypted repository secrets, never in code; secret scanning with push protection blocks commits that contain credentials |
| Dependencies | Locked in `uv.lock`; Dependabot alerts report known vulnerabilities |

## Privacy

PathSense counts and tracks anonymous people. It does no face recognition, identification,
demographics or emotion analysis, and the browser app processes video on the device
without uploading it.
