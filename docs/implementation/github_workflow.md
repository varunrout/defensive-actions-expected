# GitHub workflow

## Branch naming

Use descriptive branch names that identify the work type and scope. Examples:

```text
docs/project-documentation-pack
coach-analysis/cb-box-defence-phase2
fix/oof-coverage-validation
```

## PR hygiene

A good PR should:

- have a focused scope;
- describe what changed and why;
- list tests and checks run;
- call out limitations or follow-up work;
- avoid mixing documentation, source changes and generated artifacts unless explicitly required.

For documentation-only PRs, do not modify Python source, tests, notebooks, scripts, data files, model artifacts, generated outputs or CI configuration.

## Do not commit outputs

Do not commit generated artifacts under directories such as:

```text
outputs/
data/
models/
mlruns/
```

Also avoid committing generated report folders, local OOF files, figures, CSV exports, MLflow runs and model binaries.

## Squash-merge workflow

For small or medium PRs, squash merging keeps the main branch history readable. Before squash merge, ensure the PR title and final squash commit message describe the user-facing change.

## Cleaning local branches after merge

After the PR is merged and the local main branch is updated, remove the local topic branch:

```bash
git checkout master
git pull --ff-only
git branch -d docs/project-documentation-pack
```

If the remote branch still exists and should be removed:

```bash
git push origin --delete docs/project-documentation-pack
```
