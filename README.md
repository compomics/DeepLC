<img src="https://github.com/compomics/DeepLC/raw/master/img/deeplc_logo.png"
width="150" height="150" /> <br/><br/>

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deeplc?style=flat-square)
[![PyPI](https://img.shields.io/pypi/v/deeplc?style=flat-square)](https://pypi.org/project/deeplc/)
[![Conda](https://img.shields.io/conda/vn/bioconda/deeplc?style=flat-square)](https://bioconda.github.io/recipes/deeplc/README.html)
[![GitHub release](https://img.shields.io/github/v/release/compomics/DeepLC?include_prereleases&style=flat-square)](https://github.com/compomics/DeepLC/releases/latest/)
[![Build Status](https://img.shields.io/github/workflow/status/compomics/DeepLC/Python%20package%20test?style=flat-square)](https://github.com/compomics/DeepLC/actions?query=workflow%3A%22Python+package+test%22)
[![GitHub issues](https://img.shields.io/github/issues/compomics/DeepLC?style=flat-square)](https://github.com/compomics/DeepLC/issues)
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
- [Prediction models](#prediction-models)
- [Q&A](#Q&A)

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
### Installation

[![Download GUI](https://img.shields.io/badge/download-GUI-green?style=flat-square)](https://github.com/compomics/DeepLC/releases/latest/)

1. Download `deeplc_gui.zip` from the
[latest release](https://github.com/compomics/DeepLC/releases/latest/) and
unzip.
2. Install DeepLC GUI with `install_gui_windows.bat` or `install_gui_linux.sh`,
depending on your operating system.
3. Run DeepLC GUI by running the `deeplc_gui.jar`.


## Python package

### Installation

[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat-square)](http://bioconda.github.io/recipes/deeplc/README.html)
[![install with pip](https://img.shields.io/badge/install%20with-pip-blue.svg?style=flat-square)](http://bioconda.github.io/recipes/deeplc/README.html)
[![container](https://img.shields.io/badge/pull&nbsp;docker-biocontainers-green?style=flat-square)](https://quay.io/repository/biocontainers/deeplc)

Install with conda, using the bioconda and conda-forge channels:  
`conda install -c bioconda -c conda-forge deeplc`

Or install with pip:  
`pip install deeplc`

### Command line interface

To use the DeepLC CLI, run:

```sh
deeplc --file_pred <path/to/peptide_file.csv>
```

We highly recommend to add a peptide file with known retention times for
calibration:

```sh
deeplc --file_pred  <path/to/peptide_file.csv> --file_cal <path/to/peptide_file_with_tr.csv>
```

For an overview of all CLI arguments, run `deeplc --help`.

### Python module

Minimal example:

```python
import pandas as pd
from deeplc import DeepLC

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

```csv
seq,modifications,tr
AAGPSLSHTSGGTQSK,,12.1645
AAINQKLIETGER,6|Acetyl,34.095
AANDAGYFNDEMAPIEVKTK,12|Oxidation|18|Acetyl,37.3765
```

See
[examples/datasets](https://github.com/compomics/DeepLC/tree/master/examples/datasets)
for more examples.

## Prediction models

DeepLC comes with multiple CNN models trained on data from various experimental
settings:

| Model filename | Experimental settings | Publication |
| - | - | - |
| full_hc_dia_fixed_mods.hdf5 | Reverse phase | [Rosenberger et al. 2014](https://doi.org/10.1038/sdata.2014.31) |
| full_hc_LUNA_HILIC_fixed_mods.hdf5 | HILIC | [Spicer et al. 2018](https://doi.org/10.1016/j.chroma.2017.12.046) |
| full_hc_LUNA_SILICA_fixed_mods.hdf5 | HILIC | [Spicer et al. 2018](https://doi.org/10.1016/j.chroma.2017.12.046) |
| full_hc_PXD000954_fixed_mods.hdf5 | Reverse phase | [Rosenberger et al. 2014](https://doi.org/10.1038/sdata.2014.31) |

By default, DeepLC selects the best model based on the calibration dataset. If
no calibration is performed, the first default model is selected. Always keep
note of the used models and the DeepLC version.

## Q&A

**__Q: So DeepLC is able to predict the retention time for any modification?__**

Yes, DeepLC can predict the retention time of any modification. However, if the 
modification is **very** different from the peptides the model has seen during 
training the accuracy might not be satisfactory for you. For example, if the model
has never seen a phosphor atom before, the accuracy of the prediction is going to
be low.

**__Q: Installation fails. Why?__**

Please make sure to install DeepLC in a path that does not contain spaces. Run
the latest LTS version of Ubuntu or Windows 10. Make sure you have enough disk 
space available, surprisingly TensorFlow needs quite a bit of disk space. If
you are still not able to install DeepLC, please feel free to contact us:

Robbin.Bouwmeester@ugent.be and Ralf.Gabriels@ugent.be

**__Q: I have a special usecase that is not supported. Can you help?__**

Ofcourse, please feel free to contact us:

Robbin.Bouwmeester@ugent.be and Ralf.Gabriels@ugent.be

**__Q: DeepLC runs out of memory. What can I do?__**

You can try to reduce the batch size. DeepLC should be able to run if the batch size is low
enough, even on machines with only 4 GB of RAM.

**__Q: I have a graphics card, but DeepLC is not using the GPU. Why?__**

For now DeepLC defaults to the CPU instead of the GPU. Clearly, because you want
to use the GPU, you are a power user :-). If you want to make the most of that expensive
GPU, you need to change or remove the following line (at the top) in __deeplc.py__:

```
# Set to force CPU calculations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

Also change the same line in the function __reset_keras()__:

```
# Set to force CPU calculations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

Either remove the line or change to (where the number indicates the number of GPUs):

```
# Set to force CPU calculations
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
```

**__Q: What modification name should I use?__**

The names from unimod are used. The PSI-MS name is used by default, but the Interim name
is used as a fall-back if the PSI-MS name is not available. Please also see __unimod_to_formula.csv__
in the folder __unimod/__ for the naming of specific modifications.

**__Q: I have a modification that is not in unimod. How can I add the modification?__**

In the folder __unimod/__ there is the file __unimod_to_formula.csv__ that can be used to
add modifications. In the CSV file add a name (**that is unique and not present yet**) and
the change in atomic composition. For example:

```
Met->Hse,O,H(-2) C(-1) S(-1)
```

Make sure to use negative signs for the atoms subtracted.

**__Q: Help, all my predictions are between [0,10]. Why?__**

It is likely you did not use calibration. No problem, but the retention times for training
purposes were normalized between [0,10]. This means that you probably need to adjust the 
retention time yourselve after analysis or use a calibration set as the input.

**__Q: How does the ensemble part of DeepLC work?__**

Models within the same directory are grouped if they overlap in their name. The overlap
has to be in their full name, except for the last part of the name after a "_"-character.

The following models will be grouped:

```
full_hc_dia_fixed_mods_a.hdf5
full_hc_dia_fixed_mods_b.hdf5
```

None of the following models will not be grouped:

```
full_hc_dia_fixed_mods2_a.hdf5
full_hc_dia_fixed_mods_b.hdf5
full_hc_dia_fixed_mods_2_b.hdf5
```

**__Q: I would like to take the ensemble average of multiple models, even if they are trained on different datasets. How can I do this?__**

Feel free to experiment! Models within the same directory are grouped if they overlap in
their name. The overlap has to be in their full name, except for the last part of the 
name after a "_"-character.

The following models will be grouped:

```
model_dataset1.hdf5
model_dataset2.hdf5
```

So you just need to rename you models.