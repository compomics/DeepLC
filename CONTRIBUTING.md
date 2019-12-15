# Contributing

This document briefly describes how to contribute to
[DeepLC](https://github.com/HUPO-PSI/SpectralLibraryFormat).

## Before you begin

If you have an idea for a feature, use case to add or an approach for a bugfix,
it is best to communicate with the community by creating an issue in
[GitHub issues](https://github.com/HUPO-PSI/SpectralLibraryFormat/issues).

## How to contribute

- Fork [DeepLC](https://github.com/HUPO-PSI/SpectralLibraryFormat) onGitHub to make your changes.
- Commit and push your changes to your
[fork](https://help.github.com/articles/pushing-to-a-remote/).
- Open a
[pull request](https://help.github.com/articles/creating-a-pull-request/)
with these changes. You pull request message ideally should include:
   - A description of why the changes should be made.
   - A description of the implementation of the changes.
   - A description of how to test the changes.
- The pull request should pass all the continuous integration tests which are
  automatically run by GitHub Actions.


## Development workflow

- Main development happens on the `master` branch.

- When a new version is ready to be published:

    1. Merge into the `releases` branch.
    2. Change the version number in `setup.py` using [semantic versioning](https://semver.org/).
    3. Update the changelog (if not already done) in `CHANGELOG.md` according to [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
    4. Set a new tag with the version number, e.g. `git tag 0.1.1-dev1`
    4. Push to GitHub, with the tag: `git push --tags`

- When new commits are pushed to the `releases` branch, the following GitHub Actions are triggered:

    1. The Python package is build and published to PyPI.
    2. A zip archive is made of the `./deeplc_gui/` directory, excluding `./deeplc_gui/src` with [Zip Release](https://github.com/marketplace/actions/zip-release).
    3. A GitHub release is made with the zipped GUI files as asset and the new changes listed in `CHANGELOG.md` with [Git Release](https://github.com/marketplace/actions/git-release).
