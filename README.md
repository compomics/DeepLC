<img src="https://github.com/compomics/DeepLC/raw/master/img/deeplc_logo.png"
width="150" height="150" /> <br/><br/>

![Python](https://img.shields.io/badge/python-%3E3.6-blue?style=flat-square)
[![Build
Status](https://img.shields.io/travis/compomics/DeepLC?style=flat-square)](https://travis-ci.org/compomics/DeepLC)
[![GitHub
issues](https://img.shields.io/github/issues/compomics/DeepLC?style=flat-square)](https://github.com/compomics/DeepLC/issues)
[![GitHub](https://img.shields.io/github/license/compomics/DeepLC.svg?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0)

DeepLC: Retention time prediction for (modified) peptides using Deep Learning.

---

- [Introduction](#introduction)
- [Graphical user interface](#graphical-user-interface)
- [Python package](#python-package)
    - [Installation](#installation)
    - [Command line interface](#command-line-interface)
    - [Python module](#python-module)
- [Input files](#input-files)
---

## Introduction
DeepLC is a retention time predictor for (modified) peptides that employs Deep
Learning. It's strength lies in the fact that it can accurately predict
retention times for modified peptides, even if hasn't seen said modification
during training.

DeepLC can be run with a graphical user interface (GUI) or as a Python package.
In the latter case, DeepLC can be used from the command line, or as a python
module.

## Graphical user interface
...

## Python package
### Installation
Clone the repository and install with pip:
```
pip install .
```

### Command line interface
To use the DeepLC CLI, run:
```
deeplc --file_pred <path/to/peptide_file.csv> 
```
We highly recommend to add a peptide file with known retention times for
calibration:
```
deeplc --file_pred  <path/to/peptide_file.csv> --file_cal <path/to/peptide_file_with_tr.csv> 
```
For an overview of all CLI arguments, run `deeplc --help`.

### Pyhon module
Minimal example:
```python
import pandas as pd
from deeplc import DeepLC, FeatExtractor

peptide_file = "datasets/test_pred.csv"
calibration_file = "datasets/test_train.csv"

pep_df = pd.read_csv(peptide_file, sep=",")
pep_df['modifications'] = pep_df['modifications'].fillna("")

cal_df = pd.read_csv(calibration_file, sep=",")
cal_df['modifications'] = cal_df['modifications'].fillna("")

dlc = DeepLC()
dlc.calibrate_preds(seq_df=cal_df)
preds = dlc.make_preds(seq_df=pep_df)
```

For a more elaborate example, see
[examples/deeplc_example.py](https://github.com/compomics/DeepLC/blob/master/examples/deeplc_example.py)
.

## Input files
DeepLC expects comma-separated values (CSV) with the following columns:
- `seq`: unmodified peptide sequences
- `modifications`: MS2PIP-style formatted modifications: Every modification is
  listed as `location|name`, separated by a pipe (`|`) between the location, the
  name, and other modifications. `location` is an integer counted starting at 1
  for the first AA. 0 is reserved for N-terminal modifications, -1 for
  C-terminal modifications. `name` has to correspond to a Unimod (PSI-MS) name.
- `tr`: retention time (only required for calibration)

For example:
```
seq,modifications,tr
AAGPSLSHTSGGTQSK,,12.1645
AAINQKLIETGER,6|Acetyl,34.095
AANDAGYFNDEMAPIEVKTK,12|Oxidation|18|Acetyl,37.3765
```
