# Contributing

This document briefly describes how to contribute to
[DeepLC](https://github.com/compomics/DeepLC).

## Before you begin

If you have an idea for a feature, use case to add or an approach for a bugfix,
it is best to communicate with the community by creating an issue in
[GitHub issues](https://github.com/compomics/DeepLC/issues).

## How to contribute

- Fork [DeepLC](https://github.com/compomics/DeepLC) on GitHub to
make your changes.
- Commit and push your changes to your
[fork](https://help.github.com/articles/pushing-to-a-remote/).
- Open a
[pull request](https://help.github.com/articles/creating-a-pull-request/)
with these changes. You pull request message ideally should include:
   - A description of why the changes should be made.
   - A description of the implementation of the changes.
   - A description of how to test the changes.
- The pull request should pass all the continuous integration tests which are
  automatically run by
  [GitHub Actions](https://github.com/compomics/DeepLC/actions).


## Development workflow

- When a new version is ready to be published:

    1. Change the version number in `setup.py` using
    [semantic versioning](https://semver.org/).
    2. Update the changelog (if not already done) in `CHANGELOG.md` according to
    [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
    3. Set a new tag with the version number, e.g. `git tag v0.1.5`.
    4. Push to GitHub, with the tag: `git push; git push --tags`.

- When a new tag is pushed to (or made on) GitHub that matches `v*`, the
following GitHub Actions are triggered:

    1. The Python package is build and published to PyPI.
    2. A zip archive is made of the `./deeplc_gui/` directory, excluding
    `./deeplc_gui/src` with
    [Zip Release](https://github.com/marketplace/actions/zip-release).
    3. A GitHub release is made with the zipped GUI files as assets and the new
    changes listed in `CHANGELOG.md` with
    [Git Release](https://github.com/marketplace/actions/git-release).
    4. After some time, the bioconda package should get updated automatically.
