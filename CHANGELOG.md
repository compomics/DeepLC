# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.27] - 2021-03-16
- Ignore library messages

## [0.1.26] - 2021-03-15
- Force older library h5py for compatability

## [0.1.25] - 2021-03-12
- Library gets appended for non-calibration peptides too

## [0.1.24] - 2021-03-12
- Change the default windows install to pip instead of conda
- Change reporting of identifiers used from library

## [0.1.23] - 2021-03-04
- Log the amount of identifiers in library used

## [0.1.22] - 2021-03-04
- Publish PyPI and GitHub release

## [0.1.21] - 2021-03-04
- Add library functionality that allows for storing and retrieving predictions (without running the model)

## [0.1.20] - 2021-02-19
- Describe hyperparameters and limit CPU threads
- Additional modfications, including those exclusive to pFind
- Change calibration error to warning (since it is a warning if it is out of range...)

## [0.1.18] - 2021-01-11
- Limit CPU usage by tensorflow by connecting to n_jobs

## [0.1.17] - 2020-08-07
- Support for Python 3.8

## [0.1.16] - 2020-05-18
- Bug fix in the calibration function
- Had to order the predicted instead of observed retention times of the calibration analytes
- Thanks to @courcelm for both finding and fixing the issue

## [0.1.15] - 2020-05-15
- Different calibration function, should not contain gaps anymore
- Changed to more accurate rounding
- Changed to splitting in groups of retention time instead of groups of peptides

## [0.1.14] - 2020-02-21
- Changed default model to a different data set

## [0.1.13] - 2020-02-21
- Duplicate peptides charge fix

## [0.1.12] - 2020-02-15
- Support for charges and spaces in peprec

## [0.1.11] - 2020-02-13
- Fixes in GUI

## [0.1.10] - 2020-02-10
- Include less models in package to meet PyPI 60MB size limitation

## [0.1.9] - 2020-02-09
- Bugfix: Pass custom activation function

## [0.1.8] - 2020-02-07
- Fixed support for averaging predictions of groups of models (ensemble) when no models were passed
- New models for ensemble

## [0.1.7] - 2020-02-07
- Support for averaging predictions of groups of models (ensemble)

## [0.1.6] - 2020-01-21
- Fix the latest release

## [0.1.5] - 2020-01-21
- Spaces in paths to files and installation allowed
- References to other CompOmics tools removed in GUI

## [0.1.5] - 2020-02-13
- Fixes in GUI

## [0.1.4] - 2020-01-17
- Fix the latest release

## [0.1.3] - 2020-01-17
- Fixed the .bat installer (now uses bioconda)

## [0.1.2] - 2019-12-19
- Example files in GUI folder
- Unnecesary bat and sh for running GUI removed

## [0.1.1] - 2019-12-18
- Switch to setuptools
- Reorder publish workflow; build wheels

## [0.1.1.dev8] - 2019-12-16
- Remove xgboost dependancy

## [0.1.1.dev7] - 2019-12-16
- Use dot instead of dash in versioning for bioconda

## [0.1.1-dev6] - 2019-12-15
- Fix publish action branch specification (2)

## [0.1.1-dev5] - 2019-12-15
- Fix publish action branch specification

## [0.1.1-dev4] - 2019-12-15
- Test other trigger for publish action

## [0.1.1-dev3] - 2019-12-15
- Update documentation, specify branch in publish action

## [0.1.1-dev2] - 2019-12-15
- Add long description to setup.py

## [0.1.1-dev1] - 2019-12-15
- Initial pre-release
