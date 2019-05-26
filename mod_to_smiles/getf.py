from rdkit import Chem
from rdkit.Chem import Descriptors
from subprocess import Popen
from subprocess import PIPE
from os import remove

def rdkit_descriptors(mol,lim_f=set(["Chi0n","LabuteASA","MolLogP","MinPartialCharge","BalabanJ"])):
    ret_dict = {}
    for name,func in Descriptors.descList:
        if not name in lim_f: continue
        ret_dict[name] = func(mol)
    return(ret_dict)

def cdk_descriptors(mol,temp_f_smiles_name="tempsmiles.smi",temp_f_cdk_name="tempcdk.txt"):
    ret_dict = {}

    smiles = Chem.MolToSmiles(mol,1)
    
    temp_f_smiles = open(temp_f_smiles_name,"w")
    temp_f_smiles.write("%s temp" % smiles)
    temp_f_smiles.close()

    ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="topological"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="geometric"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="constitutional"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="electronic"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="hybrid"))

    remove(temp_f_smiles_name)
    remove(temp_f_cdk_name)

    return(ret_dict)

def call_cdk(infile="",outfile="",descriptors=""):
    cmd = "java -jar CDKDescUI-1.4.6.jar -b %s -a -t %s -o %s" % (infile,descriptors,outfile)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out = p.communicate()
    return(parse_cdk_file(outfile))

def parse_cdk_file(file):
    cdk_file = open(file).readlines()
    cols = cdk_file[0].strip().split()[1:]
    feats = cdk_file[1].strip().split()[1:]
    return(dict(zip(cols, feats)))

def getf(mol,progs=["rdkit"]):
    ret_dict = {}
    print(mol)
    mol = Chem.MolFromSmiles(mol)
    if "rdkit" in progs: ret_dict["rdkit"] = rdkit_descriptors(mol)
    if "cdk" in progs: ret_dict["cdk"] = cdk_descriptors(mol)
    return(ret_dict)


if __name__ == "__main__":
    test_smile = "N12CCC36C1CC(C(C2)=CCOC4CC5=O)C4C3N5c7ccccc76"
    test_mol = Chem.MolFromSmiles(test_smile)
    get_features(test_mol)