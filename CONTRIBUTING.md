# Contributor Guide

## Continuous Integration

Test pipelines are run on each push to a branch and on master, within Rigetti's private Gitlab.
TODO: this should also be performed publicly within GitHub Actions.

## Commit Convention

This repository uses the ESLint commit prefixing convention documented
[here](https://github.com/conventional-changelog-archived-repos/conventional-changelog-eslint/blob/master/convention.md)
with configuration shown in [package.json](./package.json).

In order to trigger version bumps (and releases to PyPI), you must use these configured commit
prefixes. That is, if you want the changes on your branch to trigger a minor version bump, then
you should prefix them with `Update:` (case sensitive) as described in `package.json`.

When the changes are merged to master, any changes visible on `master` since the most recent git
tag are what will be used to calculate the type of version bump. The tag will be committed back
to the repository automatically along with an updated `setup.py`.
