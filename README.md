<img src="https://github.com/compomics/DeepLC/raw/master/img/deeplc_logo.png"
width="150" height="150" /> <br/><br/>

[![GitHub release](https://flat.badgen.net/github/release/compomics/deeplc)](https://github.com/compomics/DeepLC/releases/latest/)
[![PyPI](https://flat.badgen.net/pypi/v/deeplc)](https://pypi.org/project/deeplc/)
[![Conda](https://img.shields.io/conda/vn/bioconda/deeplc?style=flat-square)](https://bioconda.github.io/recipes/deeplc/README.html)
[![GitHub Workflow Status](https://flat.badgen.net/github/checks/compomics/deeplc/)](https://github.com/compomics/deeplc/actions/)
[![License](https://flat.badgen.net/github/license/compomics/deeplc)](https://www.apache.org/licenses/LICENSE-2.0)
[![Twitter](https://flat.badgen.net/twitter/follow/compomics?icon=twitter)](https://twitter.com/compomics)

DeepLC: Retention time prediction for (modified) peptides using Deep Learning.

---

- [Introduction](#introduction)
- [Citation](#citation)
- [Usage](#usage)
  - [Web application](#web-application)
  - [Graphical user interface](#graphical-user-interface)
  - [Python package](#python-package)
    - [Installation](#installation)
    - [Command line interface](#command-line-interface)
    - [Python module](#python-module)
  - [Input files](#input-files)
  - [Prediction models](#prediction-models)
- [Q&A](#qa)

---

## Introduction

DeepLC is a retention time predictor for (modified) peptides that employs Deep
Learning. Its strength lies in the fact that it can accurately predict
retention times for modified peptides, even if hasn't seen said modification
during training.

DeepLC can be used through the
[web application](https://iomics.ugent.be/deeplc/),
locally with a graphical user interface (GUI), or as a Python package. In the
latter case, DeepLC can be used from the command line, or as a Python module.

## Citation

If you use DeepLC for your research, please use the following citation:
>**DeepLC can predict retention times for peptides that carry as-yet unseen modifications**  
>Robbin Bouwmeester, Ralf Gabriels, Niels Hulstaert, Lennart Martens & Sven Degroeve  
> Nature Methods 18, 1363–1369 (2021) [doi: 10.1038/s41592-021-01301-5](http://dx.doi.org/10.1038/s41592-021-01301-5)

## Usage

### Web application
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iomics.ugent.be/deeplc/)

Just go to [iomics.ugent.be/deeplc](https://iomics.ugent.be/deeplc/) and get started!


### Graphical user interface

#### In an existing Python environment (cross-platform)

1. In your terminal with Python (>=3.7) installed, run `pip install deeplc`
2. Start the GUI with the command `deeplc-gui` or `python -m deeplc.gui`

#### Standalone installer (Windows)

[![Download GUI](https://flat.badgen.net/badge/download/GUI/blue)](https://github.com/compomics/DeepLC/releases/latest/)


1. Download the DeepLC installer (`DeepLC-...-Windows-64bit.exe`) from the
[latest release](https://github.com/compomics/DeepLC/releases/latest/)
2. Execute the installer
3. If Windows Smartscreen shows a popup window with "Windows protected your PC",
click on "More info" and then on "Run anyway". You will have to trust us that
DeepLC does not contain any viruses, or you can check the source code 😉
4. Go through the installation steps
5. Start DeepLC!

![GUI screenshot](https://github.com/compomics/DeepLC/raw/img/gui-screenshot.png)


### Python package

#### Installation

[![install with bioconda](https://flat.badgen.net/badge/install%20with/bioconda/green)](http://bioconda.github.io/recipes/deeplc/README.html)
[![install with pip](https://flat.badgen.net/badge/install%20with/pip/green)](http://bioconda.github.io/recipes/deeplc/README.html)
[![container](https://flat.badgen.net/badge/pull/biocontainer/green)](https://quay.io/repository/biocontainers/deeplc)

Install with conda, using the bioconda and conda-forge channels:
`conda install -c bioconda -c conda-forge deeplc`

Or install with pip:
`pip install deeplc`

#### Command line interface

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

#### Python module

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

### Input files

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

### Prediction models

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

The table above is for an old version of DeepLC, the current version comes with:

| Model filename | Experimental settings | Publication |
| - | - | - |
| full_hc_hela_hf_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5 | Reverse phase | [Kelstrup et al. 2018](https://doi.org/10.1021/acs.jproteome.7b006021) |
| full_hc_hela_hf_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5 | Reverse phase | [Kelstrup et al. 2018](https://doi.org/10.1021/acs.jproteome.7b00602) |
| full_hc_hela_hf_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5 | Reverse phase | [Kelstrup et al. 2018](https://doi.org/10.1021/acs.jproteome.7b00602) |
| full_hc_PXD005573_mcp_cb975cfdd4105f97efa0b3afffe075cc.hdf5 | Reverse phase | [Bruderer et al. 2017](https://pubmed.ncbi.nlm.nih.gov/29070702/) |

For all the full models that can be used in DeepLC (including some TMT models!) please see:

[https://github.com/RobbinBouwmeester/DeepLCModels](https://github.com/RobbinBouwmeester/DeepLCModels)


## Q&A

**__Q: Is it required to indicate fixed modifications in the input file?__**

Yes, even modifications like carbamidomethyl should be in the input file.

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


**__Q: What does the option `dict_divider` do?__**

This parameter defines the precision to use for fast-lookup of retention times
for calibration. A value of 10 means a precision of 0.1 (and 100 a precision of
0.01) between the calibration anchor points. This parameter does not influence
the precision of the calibration, but setting it too high might mean that there
is bad selection of the models between anchor points. A safe value is usually
higher than 10.


**__Q: What does the option `split_cal` do?__**

The option `split_cal`, or split calibration, sets number of divisions of the
chromatogram for piecewise linear calibration. If the value is set to 10 the
chromatogram is split up into 10 equidistant parts. For each part the median
value of the calibration peptides is selected. These are the anchor points.
Between each anchor point a linear fit is made. This option has no effect when
the pyGAM generalized additive models are used for calibration.


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

So you just need to rename your models.
