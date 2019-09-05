"""
This code is used to test run lc_pep

For the library versions see the .yml file
"""

__author__ = "Robbin Bouwmeester"
__copyright__ = "Copyright 2019"
__credits__ = ["Robbin Bouwmeester","Prof. Lennart Martens","Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__version__ = "1.0"
__maintainer__ = "Robbin Bouwmeester"
__email__ = "Robbin.bouwmeester@ugent.be"

# Native imports
import os
import math
import time
from configparser import ConfigParser
import ast
from re import sub

# Numpy
import numpy as np

# Pandas
import pandas as pd

# TODO add file instead of hardcoded; probably in the unimod folder
three_to_one = {
        "ala" : "A",
        "arg" : "R",
        "asn" : "N",
        "asp" : "D",
        "cys" : "C",
        "glu" : "E",
        "gln" : "Q",
        "gly" : "G",
        "his" : "H",
        "ile" : "I",
        "leu" : "L",
        "lys" : "K",
        "met" : "M",
        "phe" : "F",
        "pro" : "P",
        "ser" : "S",
        "thr" : "T",
        "trp" : "W",
        "tyr" : "Y",
        "val" : "V"}

class FeatExtractor():
    def __init__(self,
                main_path=os.getcwd(),
                lib_path_mod=os.path.join(os.getcwd(),"unimod/"),
                lib_path_prot_scale=os.path.join(os.getcwd(),"expasy/"),
                lib_aa_composition=os.path.join(os.getcwd(),"aa_comp_rel.csv"),
                lib_path_smiles=os.path.join(os.getcwd(),"mod_to_smiles/"),
                split_size=7,
                verbose=True,
                include_specific_posses=[0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6,-7],
                standard_feat=True,
                add_sum_feat=True,
                ptm_add_feat=True,
                chem_descr_feat=True,
                ptm_subtract_feat=True,
                add_rolling_feat=False,
                include_unnormalized=True,
                add_comp_feat=True,
                cnn_feats=False,
                config_file=None):

        # if a config file is defined overwrite standard parameters
        if config_file:
            cparser = ConfigParser()
            cparser.read(config_file)
            lib_path_mod = cparser.get("featExtractor","lib_path_mod").strip('"')
            lib_path_prot_scale = cparser.get("featExtractor","lib_path_prot_scale").strip('"')
            lib_path_smiles = cparser.get("featExtractor","lib_path_smiles").strip('"')
            split_size = cparser.getint("featExtractor","split_size")
            verbose = cparser.getboolean("featExtractor","verbose")
            add_sum_feat = cparser.getboolean("featExtractor","add_sum_feat")
            ptm_add_feat = cparser.getboolean("featExtractor","ptm_add_feat")
            ptm_subtract_feat = cparser.getboolean("featExtractor","ptm_subtract_feat")
            chem_descr_feat = cparser.getboolean("featExtractor","chem_descr_feat")
            add_rolling_feat = cparser.getboolean("featExtractor","add_rolling_feat")
            include_unnormalized = cparser.getboolean("featExtractor","include_unnormalized")
            include_specific_posses = ast.literal_eval(cparser.get("featExtractor","include_specific_posses"))

            
        self.main_path = main_path
        self.lib_struct = self.get_chem_descr(lib_path_smiles)
        self.lib_add,self.lib_subtract = self.get_libs_mods(lib_path_mod)
        # TODO Add to config file and parser
        self.lib_aa_composition = self.get_aa_composition(lib_aa_composition)
        self.lib_aa_composition["X"] = {'C': 0}
        self.split_size = split_size
        self.libs_prop = self.get_libs_aa(lib_path_prot_scale)
        self.verbose = verbose

        self.standard_feat = standard_feat
        self.chem_descr_feat = chem_descr_feat
        self.add_sum_feat = add_sum_feat
        self.ptm_add_feat = ptm_add_feat
        self.ptm_subtract_feat = ptm_subtract_feat
        self.add_rolling_feat = add_rolling_feat
        self.cnn_feats = cnn_feats
        self.include_unnormalized = include_unnormalized
        self.include_specific_posses = include_specific_posses
        self.add_comp_feat = add_comp_feat

    def __str__(self):
        return("""
  _     ____                               __            _               _                  _             
 | |   / ___|  _ __   ___ _ __            / _| ___  __ _| |_    _____  _| |_ _ __ __ _  ___| |_ ___  _ __ 
 | |  | |     | '_ \ / _ \ '_ \   _____  | |_ / _ \/ _` | __|  / _ \ \/ / __| '__/ _` |/ __| __/ _ \| '__|
 | |__| |___  | |_) |  __/ |_) | |_____| |  _|  __/ (_| | |_  |  __/>  <| |_| | | (_| | (__| || (_) | |   
 |_____\____| | .__/ \___| .__/          |_|  \___|\__,_|\__|  \___/_/\_\\__|_|  \__,_|\___|\__\___/|_|   
              |_|        |_|                                                                              
              """)

    def get_aa_composition(self,file_loc):
        # TODO document function
        return(pd.read_csv(file_loc,index_col=0).T.to_dict())

    def count_aa(self,
                seq,
                aa_order=set(["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"])):
        """
       Count amino acids in a peptide/protein

        Parameters
        ----------
        seq : str
            peptide or protein sequence
        aa_order : set
            order and actual amino acids that will be counted, due to Python 3.6 dicts the order is preserved

        Returns
        -------
        dict
            dictionary with counts of indicated amino acids (in given order, due to Python 3.6 dicts the 
            order is preserved)
        """
        return dict(zip(aa_order,[seq.count(aa)/len(seq) for aa in aa_order]))
        # TODO below is slower... How?
        #counted_aa = Counter(seq)
        #for aa in aa_order:
        #    if aa not in counted_aa.keys():
        #        counted_aa[aa] = 0
        #[counted_aa[aa] for aa in aa_order]
        #print([counted_aa(aa) for aa in aa_order if aa in aa_order else 0.0])
        #for aa in aa_order:
        #    try: feat_vector.append(counted_aa[aa]/float(sum(counted_aa.values())))
        #    except KeyError: feat_vector.append(0)
        #return aa_order
        #return dict(zip(aa_order,feat_vector))

    def analyze_lib(self,infile_name):
        """
        Make an amino acid dictionary that map one-letter amino acid code to physicochemical properties

        Parameters
        ----------
        infile_name : str
            location of the library that maps AA to property

        Returns
        -------
        dict
            dictionary with that maps an AA to a property
        """
        # TODO add file format checking

        infile = open(infile_name)
        prop_dict = {}
        for line in infile:
            line = line.strip()
            # Skip empty lines and the tail of the file
            if len(line) == 0: continue
            aa_three,val = line.lower().split(": ")
            val = float(val)
            prop_dict[three_to_one[aa_three]] = val
        return(prop_dict)

    def split_seq(self,
                a,
                n):
        """
        Split a list (a) into multiple chunks (n)

        Parameters
        ----------
        a : list
            list to split
        n : list
            number of chunks

        Returns
        -------
        list
            chunked list
        """

        # since chunking is not alway possible do the modulo of residues
        k, m = divmod(len(a), n)
        return(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def get_libs_aa(self,path):
        """
        Read a directory with multiple AA properties and create a library of these dicts

        Parameters
        ----------
        path : str
            directory of files with AA properties

        Returns
        -------
        dictionary
            the library of AA properties, names are given based on the file name it read from
        """
        listing = os.listdir(path)
        libs_prop = {}
        for infile in listing:
            if not infile.endswith(".txt"): continue
            libs_prop["".join(infile.split(".")[:-1])] = self.analyze_lib(os.path.join(path,infile))
        return(libs_prop)

    def get_pep_scales(self,
                    idents,
                    seqs,
                    libs_prop,
                    feats,
                    splits=5,
                    include_unnormalized=True):
        """
        Calculate peptide features

        Parameters
        ----------
        idents : list
            identifier list; should correspond to seqs parameter list
        seqs : list
            peptide sequence list; should correspond to idents parameter list
        libs_prop : dict
            dictionary that is a library of features; features to calculate should be included in 
            the feats parameter list
        feats : list
            list of features to use for calculation
        splits : int
            number of splits for the peptide sequence; this is needed for single splits; cannot
            be included in initial object creation
        include_unnormalized : boolean
            include unnormalized (by peptide length) versions of features

        Returns
        -------
        pd.DataFrame
            pandas dataframe containing the features calculated for the peptide
        """

        feat_dict = {}
        
        if self.verbose: t0 = time.time()
        
        aa_order = set(three_to_one.values())
        
        for name,seq in zip(idents,seqs):
            
            # main dictionary holding identifier to feature values
            feat_dict[name] = {}
            feat_dict[name]["seq_len"] = len(seq)
            feat_dict[name].update(self.count_aa(seq,aa_order=aa_order))

            # iterate over selected features from the feature library
            for f in feats:
                try:
                    feature_seqs = [self.libs_prop[f][aa] for aa in seq]
                except:
                    seq = seq.replace("O","")
                    seq = sub("\d", "", seq)
                    feature_seqs = [self.libs_prop[f][aa] for aa in seq]

                # TODO remove hard coded feature retreival
                if self.add_rolling_feat:
                    feature_seqs = pd.Series(feature_seqs)
                    feat_dict[name]["mean_two_%s" % (f)] = feature_seqs.rolling(2).sum().mean()
                    feat_dict[name]["mean_three_%s" % (f)] = feature_seqs.rolling(3).sum().mean()
                    feat_dict[name]["max_three_%s" % (f)] = feature_seqs.rolling(3).sum().max()
                    feat_dict[name]["max_two_%s" % (f)] = feature_seqs.rolling(2).sum().max()
                    feat_dict[name]["min_three_%s" % (f)] = feature_seqs.rolling(3).sum().min()
                    feat_dict[name]["min_two_%s" % (f)] = feature_seqs.rolling(2).sum().min()

                feat_dict[name]["sum_%s" % (f)] = sum(feature_seqs)/len(feature_seqs)
                if self.include_unnormalized: feat_dict[name]["sum_%s_unnorm" % (f)] = sum(feature_seqs)
                
                # included positions if needed
                for p in self.include_specific_posses:
                    feat_dict[name]["%s_%s" % (p,f)] = feature_seqs[p]
                
                feature_seqs_split = self.split_seq(feature_seqs,splits)
                feat_dict[name].update(dict([("sum_partial_%s_%s" % (f,index+1),sum(feature_seq_split)/len(feature_seq_split)) for index,feature_seq_split in enumerate(feature_seqs_split)]))
                
                if self.include_unnormalized: feat_dict[name].update(dict([("sum_partial_%s_%s_unnorm" % (f,index+1),sum(feature_seq_split)) for index,feature_seq_split in enumerate(feature_seqs_split)]))
                
        if self.verbose: print("Time to calculate features: %s seconds" % (time.time() - t0))
        
        # transpose is needed to let rows be instances (peptides) and features columns
        return pd.DataFrame(feat_dict).T

    def get_feats(self,
                seqs,
                identifiers,
                split_size = False):
        """
        Calculate peptide features

        Parameters
        ----------
        identifiers : list
            identifier list; should correspond to seqs parameter list
        seqs : list
            peptide sequence list; should correspond to idents parameter list
        split_size : int
            split size for feature retreival; make it a boolean (False) if you want to use the initialized split size 

        Returns
        -------
        pd.DataFrame
            pandas dataframe containing the features calculated for the peptide
        """
        if not split_size: split_size = self.split_size
        feats = list(self.libs_prop.keys())
        X_feats = self.get_pep_scales(identifiers,seqs,self.libs_prop,feats,splits=split_size)
        return X_feats

    def get_chem_descr(self,directory):
        mod_df = pd.read_csv(os.path.join(directory,"mod_to_struct.csv"),index_col=0)
        return(mod_df.T.to_dict())

    def get_libs_mods(self,directory):
        """
        Make a dictionary with unimod to chemical formula

        Parameters
        ----------
        directory : str
            directory of the unimod to chemical formula mapping

        Returns
        -------
        dict
            chemical formula of a PTM when it is added
        dict
            chemical formula of a PTM when it is subtracted
        """
        # TODO replace dir with actual file...
        mod_df = pd.read_csv(os.path.join(directory,"unimod_to_formula.csv"),index_col=0)
        mod_dict = mod_df.to_dict()
        return mod_dict["formula_pos"],mod_dict["formula_neg"]

    def calc_feats_mods(self,
                        formula):
        """
        Chemical formula to atom addition/subtraction

        Parameters
        ----------
        formula : str
            chemical formula

        Returns
        -------
        list
            atom naming
        list
            number of atom added/subtracted
        """
        if not formula: 
            return [],[]
        if len(str(formula)) == 0:
            return [],[]
        if type(formula) != str:
            if math.isnan(formula):
                return [],[]
        
        new_atoms = []
        new_num_atoms = []
        for atom in formula.split(" "):
            if "(" not in atom:
                atom_symbol = atom
                num_atom = 1
            else:
                atom_symbol = atom.split("(")[0]
                num_atom = atom.split("(")[1].rstrip(")")
            new_atoms.append(atom_symbol)
            new_num_atoms.append(num_atom)
        return new_atoms,map(int,new_num_atoms)

    def get_feats_mods(self,
                    seqs,
                    mods,
                    identifiers,
                    split_size=False,
                    atoms_order = set(["H","C","N","O","P","S"]),
                    add_str="_sum",
                    subtract_mods=False):
        """
        Chemical formula to atom addition/subtraction

        Parameters
        ----------
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods : list
            naming of the mods; should correspond to seqs and identifiers
        identifiers : str
            identifiers of the peptides; should correspond to seqs and mods
        split_size : int
            overwrite the set split size if needed
        atoms_order : set
            atoms to include and the order
        add_str : str
            add this substring to feature naming
        subtract_mods : boolean
            calculate the atom that are substracted in the PTM

        Returns
        -------
        pd.DataFrame
            feature matrix for peptide PTMs
        """
        if not split_size: split_size = self.split_size
        if self.verbose: t0 = time.time()
        mod_dict = {}

        len_init = len([ao+str(spl_s) for spl_s in range(split_size) for ao in atoms_order])
        for index_name,mod,seq in zip(identifiers,mods,seqs):
            mod_dict[index_name] = dict(zip([ao+str(spl_s)+add_str for spl_s in range(split_size) for ao in atoms_order],[0]*len_init))

            if not mod: 
                continue
            if len(str(mod)) == 0:
                continue
            if type(mod) != str:
                if math.isnan(mod):
                    continue

            split_mod = mod.split("|")
            for i in range(1,len(split_mod),2):
                if subtract_mods: fill_mods,num = self.calc_feats_mods(self.lib_subtract[split_mod[i].rstrip()])
                else: fill_mods,num = self.calc_feats_mods(self.lib_add[split_mod[i].rstrip()])

                loc = int(split_mod[i-1])
                
                if loc > len(seq):
                    loc = len(seq)

                relative_loc = int(math.ceil((loc/len(seq))*split_size))-1
                for fm,n in zip(fill_mods,num):
                    if fm not in atoms_order: continue
                    mod_dict[index_name]["%s%s%s" % (fm,relative_loc,add_str)] += n
        if self.verbose: print("Time to calculate mod features: %s seconds" % (time.time() - t0))
        return pd.DataFrame(mod_dict,dtype=int).T


    def get_feats_chem_descr(self,
                            seqs,
                            mods,
                            identifiers,
                            split_size=False,
                            feat_order = ["Chi0n","LabuteASA","MolLogP","MinPartialCharge","BalabanJ"],
                            add_str="_chem_descr",
                            subtract_mods=False):
        """
        Chemical formula to atom addition/subtraction

        Parameters
        ----------
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods : list
            naming of the mods; should correspond to seqs and identifiers
        identifiers : str
            identifiers of the peptides; should correspond to seqs and mods
        split_size : int
            overwrite the set split size if needed
        atoms_order : set
            atoms to include and the order
        add_str : str
            add this substring to feature naming
        subtract_mods : boolean
            calculate the atom that are substracted in the PTM

        Returns
        -------
        pd.DataFrame
            feature matrix for peptide PTMs
        """
        if not split_size: split_size = self.split_size
        if self.verbose: t0 = time.time()
        mod_dict = {}

        len_init = len([ao+str(spl_s) for spl_s in range(split_size) for ao in feat_order])
        for index_name,mod,seq in zip(identifiers,mods,seqs):
            mod_dict[index_name] = dict(zip([ao+str(spl_s)+add_str for spl_s in range(split_size) for ao in feat_order],[0.0]*len_init))
            if not mod: 
                continue
            if len(str(mod)) == 0:
                continue
            if type(mod) != str:
                if math.isnan(mod):
                    continue

            split_mod = mod.split("|")
            for i in range(1,len(split_mod),2):
                loc = split_mod[i-1]
                if subtract_mods: fill_mods,num = self.calc_feats_mods(self.lib_subtract[split_mod[i].rstrip()])
                else: fill_mods,num = self.calc_feats_mods(self.lib_add[split_mod[i].rstrip()])

                chem_descr = self.lib_struct[split_mod[i].rstrip()]
                relative_loc = int(math.ceil((int(loc)/len(seq))*split_size))-1
                for f in feat_order:
                    mod_dict[index_name]["%s%s%s" % (f,relative_loc,add_str)] += chem_descr[f]
        if self.verbose: print("Time to calculate mod features: %s seconds" % (time.time() - t0))
        return pd.DataFrame(mod_dict).T

    def get_comp_change_mods(self,
                            seqs,
                            mods,
                            identifiers,
                            positions=[0,1,2,3,4,-1,-2,-3,-4,-5],
                            atom_count=["C","H","N","O","S","P"]):
        # TODO split up function into multiple functions 
        # TODO add documentation for this function
        composition_dict = {}
        feat_dict = {}
        for ident,mod,seq in zip(identifiers,mods,seqs):
            # TODO to speed up maybe use specific lexicon to determine location C0, H0 ... C7 H7
            peptide_comp = [dict(zip(atom_count,[0]*len(atom_count))) for aa in seq]
            mod_comp = [dict(zip(atom_count,[0]*len(atom_count))) for aa in seq]
            pos_atoms = list(mod_comp[0].keys())

            for i,aa in enumerate(seq):
                if aa not in self.lib_aa_composition.keys(): 
                    continue
                peptide_comp_temp = self.lib_aa_composition[aa]
                for k,v in peptide_comp_temp.items():
                    try: peptide_comp[i][k] = v
                    except KeyError: continue
            # TODO too many errors to catch? Just test some differnt ones and see what the outcome is
            try:
                split_mod = mod.split("|")
            except:
                split_mod = []

            # TODO below very slow with modification
            for i in range(1,len(split_mod),2):
                split_mod[i-1] = int(split_mod[i-1])
                # TODO make this a dictionary call function
                fill_mods,fill_num = self.calc_feats_mods(self.lib_add[split_mod[i].rstrip()])
                subtract_mods,subtract_num = self.calc_feats_mods(self.lib_subtract[split_mod[i].rstrip()])
                
                 # TODO move to function, isnt there already a function for this?
                for atom,atom_change in zip(fill_mods,fill_num):
                    try:
                        mod_comp[split_mod[i-1]-1][atom] += atom_change
                    except KeyError:
                        print("Going to skip the following atom in modification: %s" % (split_mod[i]))
                    except IndexError:
                        print("Index does not exist for: ",atom,atom_change,ident,mod,seq)

                for atom,atom_change in zip(subtract_mods,subtract_num):
                    try:
                        mod_comp[split_mod[i-1]-1][atom] -= atom_change
                    except KeyError:
                        print("Going to skip the following atom in modification: %s" % (split_mod[i]))
                    except IndexError:
                        print("Index does not exist for: ",atom,atom_change,ident,mod,seq)
            
            peptide_comp_df = pd.DataFrame(peptide_comp)
            mod_comp_df = pd.DataFrame(mod_comp)
            # TODO is this adding step slow? See specific lexicon comment 
            combined_comp_df = peptide_comp_df+mod_comp_df
            summed_comp = dict(combined_comp_df.sum())
            sum_comp = sum(summed_comp.values())
            
            feat_dict[ident] = {}

            feat_dict[ident].update(dict([[k+"_tot",v]for k,v in summed_comp.items()]))
            feat_dict[ident].update(dict([[k+"_tot_norm",v/sum_comp]for k,v in summed_comp.items()]))

            for p in positions:
                feat_dict[ident].update(dict([[k+"_"+str(p),v] for k,v in dict(combined_comp_df.iloc[p]).items()]))

            index_splits = self.split_seq(combined_comp_df.index,self.split_size)
            
            for i_part,split_part in enumerate(index_splits):
                split_part_df = combined_comp_df.loc[split_part]
                split_part_df_sum = split_part_df.sum().sum()
                split_part_df_norm = split_part_df/split_part_df_sum
                for atom in split_part_df.columns:
                    feat_dict[ident]["part_%s_atom_%s" % (i_part,atom)] = split_part_df[atom].sum()
                    feat_dict[ident]["part_%s_atom_%s_norm" % (i_part,atom)] = split_part_df_norm[atom].sum()
            
            for pa in pos_atoms:
                feat_dict[ident]["rolling_2_atom_%s_max" % (pa)] = pd.Series([v[pa] for v in peptide_comp]).rolling(2).sum().max()
                feat_dict[ident]["rolling_2_atom_%s_min" % (pa)] = pd.Series([v[pa] for v in peptide_comp]).rolling(2).sum().min()
                feat_dict[ident]["rolling_3_atom_%s_max" % (pa)] = pd.Series([v[pa] for v in peptide_comp]).rolling(3).sum().max()
                feat_dict[ident]["rolling_3_atom_%s_min" % (pa)] = pd.Series([v[pa] for v in peptide_comp]).rolling(3).sum().min()
        
        return(pd.DataFrame(feat_dict).T)

    def encode_atoms(self,
                    seqs,
                    mods_all,
                    indexes,
                    padding_length=60,
                    positions=set([0,1,2,3,-1,-2,-3,-4]),
                    sum_mods=2,
                    dict_index_pos={'C' : 0,
                                'H' : 1,
                                'N' : 2,
                                'O' : 3,
                                'S' : 4,
                                'P' : 5},
                    dict_index_all={'C' : 0,
                                'H' : 1,
                                'N' : 2,
                                'O' : 3,
                                'S' : 4,
                                'P' : 5},
                    dict_index={'C' : 0,
                                'H' : 1,
                                'N' : 2,
                                'O' : 3,
                                'S' : 4,
                                'P' : 5}):
        
        ret_list = {}
        ret_list_sum = {}
        ret_list_all = {}
        ret_list_pos = {}

        #seqs = seqs_df["seq"]
        #indexes = seqs_df.index
        #mods_all = seqs_df["modifications"]
        
        for row_index,seq,mods in zip(indexes,seqs,mods_all):
            seq_len = len(seq)
            if seq_len > padding_length: continue
            
            padding = "".join(["X"]*(padding_length-len(seq)))
            seq = seq+padding
            
            matrix = np.zeros((len(seq),len(dict_index.keys())),dtype=np.float16)
            matrix_sum = np.zeros((int(len(seq)/sum_mods),len(dict_index.keys())),dtype=np.float16)
            matrix_all = np.zeros(len(dict_index_all.keys())+1,dtype=np.float32)
            matrix_pos = np.zeros((len(positions),len(dict_index.keys())),dtype=np.float16)

            matrix_all[len(dict_index_all.keys())] = len(seq)
            
            for index,aa in enumerate(seq):
                if aa == "X": break
                index_sum = int(index/sum_mods)
                
                for atom,val in self.lib_aa_composition[aa].items():
                    matrix[index,dict_index[atom]] = val
                    matrix_sum[index_sum,dict_index[atom]] += val
                    matrix_all[dict_index_all[atom]] += val
                    
                    if index in positions:
                        matrix_pos[index,dict_index_pos[atom]] = val
                    elif index-seq_len in positions:
                        matrix_pos[index-seq_len,dict_index_pos[atom]] = val
            
            if len(mods) == 0:
                ret_list[row_index] = {"index_name":row_index,"matrix":matrix}
                ret_list_sum[row_index] = {"index_name":row_index,"matrix_sum":matrix_sum}
                ret_list_all[row_index] = {"index_name":row_index,"matrix_all":matrix_all}
                ret_list_pos[row_index] = {"index_name":row_index,"pos_matrix":matrix_pos.flatten()}
                continue
            

            mods = mods.split("|")
            for i in range(1,len(mods),2):
                mod = mods[i]
                try:
                    fill_mods,fill_num = self.calc_feats_mods(self.lib_add[mod])
                    subtract_mods,subtract_num = self.calc_feats_mods(self.lib_subtract[mod])
                except KeyError:
                    print("Going to skip the following atom in modification: %s" % (split_mod[i]))
                
                loc = int(mods[i-1])-1
                if loc > len(seq):
                    # TODO display error
                    loc = len(seq)-1
                    
                for atom,atom_change in zip(fill_mods,fill_num):
                    try:
                        
                        matrix[loc,dict_index[atom]] += atom_change
                        matrix_all[dict_index_all[atom]] += val
                        if loc in positions:
                            matrix_pos[loc,dict_index_pos[atom]] += val
                        elif loc-len(seq) in positions:
                            matrix_pos[loc-len(seq),dict_index_pos[atom]] += val
                            
                    except KeyError:
                        print("Going to skip the following atom in modification: %s" % (split_mod[i]))
                    except IndexError:
                        print("Index does not exist for: ",atom,atom_change,ident,mod,seq)
                
                for atom,atom_change in zip(subtract_mods,subtract_num):
                    try:
                        
                        matrix[loc,dict_index[atom]] -= atom_change
                        matrix_all[dict_index_all[atom]] -= val
                        if loc in positions:
                            matrix_pos[loc,dict_index_pos[atom]] -= val
                        elif loc-len(seq) in positions:
                            matrix_pos[loc-len(seq),dict_index_pos[atom]] -= val
                            
                    except KeyError:
                        print("Going to skip the following atom in modification: %s" % (split_mod[i]))
                    except IndexError:
                        print("Index does not exist for: ",atom,atom_change,ident,mod,seq)
                        
            ret_list[row_index] = {"index_name":row_index,"matrix":matrix}
            ret_list_sum[row_index] = {"index_name":row_index,"matrix_sum":matrix_sum}
            ret_list_all[row_index] = {"index_name":row_index,"matrix_all":matrix_all}
            ret_list_pos[row_index] = {"index_name":row_index,"pos_matrix":matrix_pos.flatten()}
        
        return pd.DataFrame.from_dict(ret_list).T, pd.DataFrame.from_dict(ret_list_sum).T, pd.DataFrame.from_dict(ret_list_pos).T, pd.DataFrame.from_dict(ret_list_all).T


    def encode_atoms_only(self,
                    seqs,
                    mods,
                    identifiers,
                    padding_length=60,
                    dict_index={'C' : 0,
                                'H' : 1,
                                'N' : 2,
                                'O' : 3,
                                'S' : 4,
                                'P' : 5}):
        ret_list = {}
        for row_index,seq,modifs in zip(identifiers,seqs,mods):

            if len(seq) > padding_length: continue
            padding = "".join(["X"]*(padding_length-len(seq)))
            seq = seq+padding
            matrix = np.zeros((len(seq),len(dict_index.keys())),dtype=np.int16)
            for index,aa in enumerate(seq):
                #if aa == "X": break
                for atom,val in self.lib_aa_composition[aa].items():
                    if val == 0: continue
                    matrix[index,dict_index[atom]] = val
            
            if len(modifs) == 0:
                ret_list[row_index] = {"index_name":row_index,"matrix":matrix}
                continue
                    
            modifs = modifs.split("|")
            for i in range(1,len(modifs),2):
                mod = modifs[i]
                fill_mods,fill_num = self.calc_feats_mods(self.lib_add[mod])
                subtract_mods,subtract_num = self.calc_feats_mods(self.lib_subtract[mod])
                
                loc = int(modifs[i-1])-1
                if loc > len(seq):
                    loc = len(seq)-1
                    print("Modification out of index: %s,%s,%s,%s" % (row_index,seq,loc,mod))
                    
                for atom,atom_change in zip(fill_mods,fill_num):
                    try:
                        matrix[loc,dict_index[atom]] += atom_change
                    except KeyError:
                        print("Going to skip the following atom in modification: %s" % (split_mod[i]))
                    except IndexError:
                        print("Index does not exist for: ",atom,atom_change,ident,mod,seq)
                
                for atom,atom_change in zip(subtract_mods,subtract_num):
                    try:
                        matrix[loc,dict_index[atom]] -= atom_change
                    except KeyError:
                        print("Going to skip the following atom in modification: %s" % (split_mod[i]))
                    except IndexError:
                        print("Index does not exist for: ",atom,atom_change,ident,mod,seq)
                        
            ret_list[row_index] = {"index_name":row_index,"matrix":matrix}
        
        return pd.DataFrame(ret_list).T

    def encode_atoms_positions(self,
                              seqs,
                              mods,
                              identifiers,
                              positions=[0,1,2,3,-1,-2,-3,-4],                            
                              dict_index={'C' : 0,
                                          'H' : 1,
                                          'N' : 2,
                                          'O' : 3,
                                          'S' : 4,
                                          'P' : 5}):
        ret_list = {}
        for row_index,seq,modifs in zip(identifiers,seqs,mods):
            matrix = np.zeros((len(positions),len(dict_index.keys())),dtype=np.int16)
            for index in positions:
                for atom,val in self.lib_aa_composition[seq[index]].items():
                    if val == 0: continue
                    matrix[index,dict_index[atom]] = val
            
            if len(modifs) == 0:
                ret_list[row_index] = {"index_name":row_index,"pos_matrix":matrix.flatten()}
                continue
                    
            modifs = modifs.split("|")
            for i in range(1,len(modifs),2):
                mod = modifs[i]
                fill_mods,fill_num = self.calc_feats_mods(self.lib_add[mod])
                subtract_mods,subtract_num = self.calc_feats_mods(self.lib_subtract[mod])
                
                loc = int(modifs[i-1])-1
                if loc > len(seq):
                    loc = len(seq)-1
                if loc < 0:
                    loc = loc+len(seq)
                
                if loc not in positions:
                    continue
                        
                for atom,atom_change in zip(fill_mods,fill_num):
                    try:
                        matrix[loc,dict_index[atom]] += atom_change
                    except KeyError:
                        print("Going to skip the following atom in modification: %s" % (split_mod[i]))
                    except IndexError:
                        print("Index does not exist for: ",atom,atom_change,ident,mod,seq)
                
                for atom,atom_change in zip(subtract_mods,subtract_num):
                    try:
                        matrix[loc,dict_index[atom]] -= atom_change
                    except KeyError:
                        print("Going to skip the following atom in modification: %s" % (split_mod[i]))
                    except IndexError:
                        print("Index does not exist for: ",atom,atom_change,ident,mod,seq)
                        
            ret_list[row_index] = {"index_name":row_index,"pos_matrix":matrix.flatten()}
        
        return pd.DataFrame(ret_list).T

        

    def full_feat_extract(self,
                        seqs,
                        mods,
                        identifiers):
        """
        Extract all features we can extract... Probably the function your want to call by default

        Parameters
        ----------
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods : list
            naming of the mods; should correspond to seqs and identifiers
        identifiers : str
            identifiers of the peptides; should correspond to seqs and mods

        Returns
        -------
        pd.DataFrame
            feature matrix
        """
        if self.verbose: t0 = time.time()

        if self.standard_feat:
            if self.verbose: print("Extracting standard features")
            X = self.get_feats(seqs,identifiers,split_size=self.split_size)
        if self.add_comp_feat:
            if self.verbose: print("Extracting compositional features")
            X_comp = self.get_comp_change_mods(seqs,mods,identifiers)
        if self.add_sum_feat:
            if self.verbose: print("Extracting compositional sum features for modifications")
            X_feats_sum = self.get_feats_mods(seqs,mods,identifiers,split_size=1,add_str="_sum")
        if self.ptm_add_feat:
            if self.verbose: print("Extracting compositional add features for modifications")
            X_feats_add = self.get_feats_mods(seqs,mods,identifiers,split_size=self.split_size,add_str="_add")
        if self.ptm_subtract_feat:
            if self.verbose: print("Extracting compositional subtract features for modifications")
            X_feats_neg = self.get_feats_mods(seqs,mods,identifiers,split_size=self.split_size,add_str="_subtract",subtract_mods=True)
        if self.chem_descr_feat:
            if self.verbose: print("Extracting chemical descriptors features for modifications")
            X_feats_chem_descr = self.get_feats_chem_descr(seqs,mods,identifiers,split_size=3,add_str="_chem_desc")
        if self.cnn_feats:
            if self.verbose: print("Extracting CNN features")
            X_cnn,X_sum,X_cnn_pos,X_cnn_count = self.encode_atoms(seqs,mods,identifiers)
            X_cnn = pd.concat([X_cnn,X_sum,X_cnn_pos,X_cnn_count],axis=1)

        if self.cnn_feats:
            try: X = pd.concat([X,X_cnn],axis=1)
            except: X = X_cnn
        if self.add_comp_feat:
            try: X = pd.concat([X,X_comp],axis=1)
            except: X = X_comp
        if self.add_sum_feat:
            try: X = pd.concat([X,X_feats_sum],axis=1)
            except: X = X_feats_sum
        if self.ptm_add_feat:
            try: X = pd.concat([X,X_feats_add],axis=1)
            except: X = X_feats_add
        if self.ptm_subtract_feat:
            try: X = pd.concat([X,X_feats_neg],axis=1)
            except: X = X_feats_neg
        if self.chem_descr_feat:
            try: X = pd.concat([X,X_feats_chem_descr],axis=1)
            except: X = X_feats_chem_descr

        if self.verbose: print("Time to calculate all features: %s seconds" % (time.time() - t0))
        return X

def main(verbose=True):
    f_extractor = FeatExtractor(config_file="config.ini")
    df = pd.read_csv("parse_pride/seqs_exp.csv")
    df.index = ["Pep_"+str(dfi) for dfi in df.index]
    print(f_extractor.full_feat_extract(df["seq"],df["modifications"],df.index))

if __name__ == "__main__":
    main()