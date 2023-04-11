import pandas as pd
from deeplc import DeepLC
from matplotlib import pyplot as plt

def main():
    dlc = DeepLC() 

    dlc.calibrate_preds(infile="msms.txt")
    preds = dlc.make_preds(infile="msms.txt")

    psm_list = read_file(infile)
    if "msms" in infile and ".txt" in infile:
        mapper = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "unimod/map_mq_file.csv"),index_col=0)["value"].to_dict()
        psm_list.rename_modifications(mapper)

    plt.scatter([psm.retention_time for psm in psm_list],preds)

    print(preds)
    plt.savefig("test.png")

if __name__ == "__main__":
    main()