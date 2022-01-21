import pandas as pd
from deeplc import DeepLC
from matplotlib import pyplot as plt

def main():
    peptide_file = "temp_data/PXD005573_mcp.csv"
    calibration_file = "temp_data/PXD005573_mcp.csv"

    pep_df = pd.read_csv(peptide_file, sep=",")
    pep_df['modifications'] = pep_df['modifications'].fillna("")

    cal_df = pd.read_csv(calibration_file, sep=",")
    cal_df['modifications'] = cal_df['modifications'].fillna("")
    
    pep_df = pep_df.sample(50)
    cal_df = cal_df.sample(50)

    mods = ["C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_arabidopsis_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5",
            #"C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_PXD005573_mcp_1fd8363d9af9dcad3be7553c39396960.hdf5",
            #"C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_PXD005573_mcp_8c22d89667368f2f02ad996469ba157e.hdf5",
            #"C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_PXD005573_mcp_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_PXD008783_median_calibrate_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_PXD008783_median_calibrate_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_PXD008783_median_calibrate_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_dia_fixed_mods_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_dia_fixed_mods_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_dia_fixed_mods_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_phospho_kai_li_5ee8aaa41d387bfffb8cda966348937c.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_phospho_kai_li_8c488fed5e0d0b07cf217fe3c30e55c6.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_phospho_kai_li_f3c75e74dd7b16180edf6f6f0d78a4a6.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_prosit_ptm_2020_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_prosit_ptm_2020_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_prosit_ptm_2020_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_tmt_data_consensus_ticnum_filtered_5ee8aaa41d387bfffb8cda966348937c.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_tmt_data_consensus_ticnum_filtered_8c488fed5e0d0b07cf217fe3c30e55c6.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_yeast_120min_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_yeast_120min_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_yeast_120min_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_arabidopsis_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_arabidopsis_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5",            
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_hela_hf_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_hela_hf_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_hela_hf_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_hela_lumos_1h_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_hela_lumos_1h_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_hela_lumos_1h_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_hela_lumos_2h_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_hela_lumos_2h_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_hela_lumos_2h_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_pancreas_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_pancreas_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_pancreas_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_plasma_lumos_1h_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_plasma_lumos_1h_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_plasma_lumos_1h_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_plasma_lumos_2h_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_plasma_lumos_2h_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_plasma_lumos_2h_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_yeast_60min_psms_aligned_1fd8363d9af9dcad3be7553c39396960.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_yeast_60min_psms_aligned_8c22d89667368f2f02ad996469ba157e.hdf5",
            "C:/Users/robbin/Documents/Github/DeepLCModels/full_hc_yeast_60min_psms_aligned_cb975cfdd4105f97efa0b3afffe075cc.hdf5"]
    
    dlc = DeepLC(write_library=False,
                 use_library="",
                 pygam_calibration=True,
                 deepcallc_mod=True,
                 path_model=mods,
                 reload_library=False)

    dlc.calibrate_preds(seq_df=cal_df)
    preds = dlc.make_preds(seq_df=cal_df)

    plt.scatter(cal_df["tr"],preds)
    plt.show()

if __name__ == "__main__":
    main()