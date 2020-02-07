# Imports
import pandas as pd
from matplotlib import pyplot as plt
from deeplc import DeepLC, FeatExtractor

# Input files
peptide_file = "examples/datasets/test_train.csv"

# Read the input data to make predictions for
df = pd.read_csv(peptide_file, sep=",")

# Unmodified peptides in modifications column should be empty strings, not nan
df = df.fillna("")

# Generate some identifiers, any kind of identifiers will do
df.index = ["Pep_"+str(dfi) for dfi in df.index]

# Make a feature extraction object.
# This step can be skipped if you want to use the default feature extraction
# settings. In this example we will use a model that does not use RDKit features
# so we skip the chemical descriptor making procedure.

#f_extractor = FeatExtractor(chem_descr_feat=False,verbose=False)
f_extractor = FeatExtractor(
    add_sum_feat=False,
    ptm_add_feat=False,
    ptm_subtract_feat=False,
    standard_feat=False,
    chem_descr_feat=False,
    add_comp_feat=False,
    cnn_feats=True,
    verbose=False
)
# Initiate a DeepLC instance that will perform the calibration and predictions
dlc = DeepLC(
    path_model=["deeplc/mods/full_hc_dia_fixed_mods_a.hdf5","deeplc/mods/full_hc_dia_fixed_mods_b.hdf5"],
    cnn_model=True,
    f_extractor=f_extractor,
    verbose=False
)

# To demonstrate DeepLC's callibration, we'll induce some an artificial
# transformation into the retention times
df["tr"] = df["tr"]**0.85

# Calibrate the original model based on the new retention times
dlc.calibrate_preds(seq_df=df)

# Make predictions; calibrated and uncalibrated
preds_cal = dlc.make_preds(seq_df=df)
preds_uncal = dlc.make_preds(seq_df=df, calibrate=False)

# Compare calibrated and uncalibrated predictions
#print("Predictions (calibrated): ", preds_cal)
#print("Predictions (uncalibrated): ", preds_uncal)

plt.scatter(df["tr"],preds_cal,label="Calibrated",s=1)
plt.scatter(df["tr"],preds_uncal,label="Uncalibrated",s=1)
plt.legend()
plt.savefig('deeplc_calibrated_vs_uncalibrated.png')
