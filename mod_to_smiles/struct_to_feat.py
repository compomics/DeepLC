import pandas as pd

#cheminformatics
from getf import getf
import rdkit

def get_chem_descr():
    mod_df = pd.read_csv("mod_to_smiles.txt",index_col=0,sep="|")
    chem_descr_dict = {}
    for index_name,smiles in mod_df.iterrows():
        chem_descr_dict[index_name] = getf(smiles["SMILES"])["rdkit"]
    return(chem_descr_dict)

if __name__ == "__main__":
    pd.DataFrame(get_chem_descr()).T.to_csv("mod_to_struct.csv")

    df = pd.read_csv("mod_to_struct.csv",index_col=0)
    print(df.T.to_dict())