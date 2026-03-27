# Workflows

## Automatic Tag

Automatically tags and creates release whenever the pyproject version is increased

## Codecov

Creates a code coverage report and uploads to https://app.codecov.io/gh/henry-wallace-phys/MaCh3SbiTools
whenever a PR to `main` is made

## Docs

Generates documentation when a PR is accepted and uploads to https://henry-wallace-phys.github.io/MaCh3SbiTools/

## label

Labels PRs based on the components in MaCh3SBITools they've changed. The list can be found in [labeler.yml](../labeler.yml)

## mypy

Runs mypy checks. Done in python 3.11, 3.12 and 3.13 for completeness

## pr_title_check

Makes sure PR titles are labelled correctly. The full list is in [pr-title-checker-config.json](../pr-title-checker-config.json)

## pytest

Runs the full pytest suite on the repo for Windows, Mac-OS and ubuntu. 3.13 is omitted in Windows AND Mac due to depedency
issues

## ruff

Runs the Ruff linter over the entire package
