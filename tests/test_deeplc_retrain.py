import pandas as pd
from deeplc import DeepLC
from matplotlib import pyplot as plt

def main():
    peptide_file = "examples/datasets/test_pred.csv"
    calibration_file = "examples/datasets/test_train.csv"

    pep_df = pd.read_csv(peptide_file, sep=",")
    pep_df['modifications'] = pep_df['modifications'].fillna("")

    cal_df = pd.read_csv(calibration_file, sep=",")
    cal_df['modifications'] = cal_df['modifications'].fillna("")

    dlc = DeepLC(write_library=False,
                 deeplc_retrain=True,
                 reload_library=False)
                 #write_library=True,
                 #use_library="lib.csv",
                 #reload_library=True)
    dlc.calibrate_preds(seq_df=cal_df)
    preds = dlc.make_preds(seq_df=cal_df)

    plt.scatter(cal_df["tr"],preds)
    plt.show()

if __name__ == "__main__":
    main()