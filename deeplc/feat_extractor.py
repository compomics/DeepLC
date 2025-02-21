"""
This code is used to extract features for peptides used in DeepLC (or seperately
if required).

For the library versions see the .yml file
"""

__author__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__credits__ = [
    "Robbin Bouwmeester",
    "Ralf Gabriels",
    "Arthur Declercq",
    "Prof. Lennart Martens",
    "Sven Degroeve",
]
__license__ = "Apache License, Version 2.0"
__maintainer__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__email__ = ["Robbin.Bouwmeester@ugent.be", "Ralf.Gabriels@ugent.be"]

# Native imports
import os
import math
import time
from configparser import ConfigParser
import ast
from re import sub
import logging

import numpy as np
import pandas as pd
from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from pyteomics import mass

logger = logging.getLogger(__name__)


class FeatExtractor:
    """
    Place holder, fill later

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(
        self,
        main_path=os.path.dirname(os.path.realpath(__file__)),
        lib_path_mod=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "unimod/"
        ),
        lib_aa_composition=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "aa_comp_rel.csv"
        ),
        split_size=7,
        verbose=True,
        include_specific_posses=[0, 1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6, -7],
        add_sum_feat=False,
        ptm_add_feat=False,
        ptm_subtract_feat=False,
        add_rolling_feat=False,
        include_unnormalized=True,
        add_comp_feat=False,
        cnn_feats=True,
        ignore_mods=False,
        config_file=None,
    ):
        # if a config file is defined overwrite standard parameters
        if config_file:
            cparser = ConfigParser()
            cparser.read(config_file)
            lib_path_mod = cparser.get("featExtractor", "lib_path_mod").strip('"')
            split_size = cparser.getint("featExtractor", "split_size")
            verbose = cparser.getboolean("featExtractor", "verbose")
            add_sum_feat = cparser.getboolean("featExtractor", "add_sum_feat")
            ptm_add_feat = cparser.getboolean("featExtractor", "ptm_add_feat")
            ptm_subtract_feat = cparser.getboolean("featExtractor", "ptm_subtract_feat")
            add_rolling_feat = cparser.getboolean("featExtractor", "add_rolling_feat")
            include_unnormalized = cparser.getboolean(
                "featExtractor", "include_unnormalized"
            )
            include_specific_posses = ast.literal_eval(
                cparser.get("featExtractor", "include_specific_posses")
            )

        self.main_path = main_path

        # Get the atomic composition of AAs
        self.lib_aa_composition = self.get_aa_composition(lib_aa_composition)

        self.split_size = split_size
        self.verbose = verbose

        self.add_sum_feat = add_sum_feat
        self.ptm_add_feat = ptm_add_feat
        self.ptm_subtract_feat = ptm_subtract_feat
        self.add_rolling_feat = add_rolling_feat
        self.cnn_feats = cnn_feats
        self.include_unnormalized = include_unnormalized
        self.include_specific_posses = include_specific_posses
        self.add_comp_feat = add_comp_feat
        self.ignore_mods = ignore_mods

    def __str__(self):
        return """
  _____                  _      _____       __           _               _                  _
 |  __ \                | |    / ____|     / _|         | |             | |                | |
 | |  | | ___  ___ _ __ | |   | |   ______| |_ ___  __ _| |_    _____  _| |_ _ __ __ _  ___| |_ ___  _ __
 | |  | |/ _ \/ _ \ '_ \| |   | |  |______|  _/ _ \/ _` | __|  / _ \ \/ / __| '__/ _` |/ __| __/ _ \| '__|
 | |__| |  __/  __/ |_) | |___| |____     | ||  __/ (_| | |_  |  __/>  <| |_| | | (_| | (__| || (_) | |
 |_____/ \___|\___| .__/|______\_____|    |_| \___|\__,_|\__|  \___/_/\_\\__|_|  \__,_|\___|\__\___/|_|
                  | |
                  |_|
        """

    def get_aa_composition(self, file_loc):
        """
        Read amino acid atomic composition and return a dictionary

        Parameters
        ----------
        file_loc : str
            location of the (csv) file that contains the atomic compositions of AAs. The first column must contain
            the one-letter AA code. The remaining columns contain counts for each atom (each atom in seperate
            column). An example is:

                aa,C,H,N,O,S
                A,1,2,0,0,0
                R,4,9,3,0,0
                N,2,3,1,1,0

        Returns
        -------
        dict
            dictionary that goes from one-letter AA to atomic composition
        """
        return pd.read_csv(file_loc, index_col=0).T.to_dict()

    def split_seq(self, a, n):
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
        return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    def calc_feats_mods(self, formula):
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
            return [], []
        if len(str(formula)) == 0:
            return [], []
        if not isinstance(formula, str):
            if math.isnan(formula):
                return [], []

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
            new_num_atoms.append(int(num_atom))
        return new_atoms, new_num_atoms

    def get_feats_mods(
        self,
        seqs,
        mods,
        identifiers,
        split_size=False,
        atoms_order=set(["H", "C", "N", "O", "P", "S"]),
        add_str="_sum",
        subtract_mods=False,
    ):
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
        object :: pd.DataFrame
            feature matrix for peptide PTMs
        """
        if not split_size:
            split_size = self.split_size
        if self.verbose:
            t0 = time.time()
        mod_dict = {}
        look_up_mod_subtract = {}
        look_up_mod_add = {}

        len_init = len(
            [ao + str(spl_s) for spl_s in range(split_size) for ao in atoms_order]
        )
        for index_name, mod, seq in zip(identifiers, mods, seqs):
            mod_dict[index_name] = dict(
                zip(
                    [
                        ao + str(spl_s) + add_str
                        for spl_s in range(split_size)
                        for ao in atoms_order
                    ],
                    [0] * len_init,
                )
            )

            if not mod:
                continue
            if len(str(mod)) == 0:
                continue
            if not isinstance(mod, str):
                if math.isnan(mod):
                    continue

        if self.verbose:
            logger.debug(
                "Time to calculate mod features: %s seconds" % (time.time() - t0)
            )
        df_ret = pd.DataFrame(mod_dict, dtype=int).T
        del mod_dict
        return df_ret

    def encode_atoms(
        self,
        psm_list,
        indexes,
        charges=[],
        predict_ccs=False,
        padding_length=60,
        positions=set([0, 1, 2, 3, -1, -2, -3, -4]),
        positions_pos=set([0, 1, 2, 3]),
        positions_neg=set([-1, -2, -3, -4]),
        sum_mods=2,
        dict_aa={
            "K": 0,
            "R": 1,
            "P": 2,
            "T": 3,
            "N": 4,
            "A": 5,
            "Q": 6,
            "V": 7,
            "S": 8,
            "G": 9,
            "I": 10,
            "L": 11,
            "C": 12,
            "M": 13,
            "H": 14,
            "F": 15,
            "Y": 16,
            "W": 17,
            "E": 18,
            "D": 19,
        },
        dict_index_pos={"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5},
        dict_index_all={"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5},
        dict_index={"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5},
    ):
        """
        Extract all features we can extract... Probably the function you want to call by default

        Parameters
        ----------
        seqs : list
            peptide sequence list; should correspond to mods and identifiers
        mods_all : list
            naming of the mods; should correspond to seqs and identifiers
        indexes : list
            identifiers of the peptides; should correspond to seqs and mods
        padding_length : int
            indicates padding length with 'X'-characters. Shorter sequences are padded. Longer sequences
            are sliced shorter (C-terminal > than padding length will be missing)
        positions : list
            list of positions to include separately, for the C-terminus
            provide negative indices
        sum_mods : int
            value that is used to feed the second head of cerberus with summed information, for example,
            a value of 2 would sum the composition of each 2 AAs (i.e. 1+2, 3+4, 5+6 ...)
        dict_index_pos : dict
            index position of atom for positional features
        dict_index_all : dict
            index position of atom for overall compositional features
        dict_index : dict
            index position of atom for compositional features for the whole peptide (each position)
        charges : list
            optional list with charges, keep empty if these will not effect the predicted value

        Returns
        -------
        object :: pd.DataFrame
            feature matrix (np.matrix) of all positions (up till padding length)
        object :: pd.DataFrame
            feature matrix (np.matrix) of summed positions (up till padding length / sum_mods)
        object :: pd.DataFrame
            feature matrix (np.matrix) of specific positions (from positions argument)
        object :: pd.DataFrame
            feature matrix (np.matrix) of summed composition
        """

        # Local helper to ensure each unique warning is logged only once.
        logged_warnings = set()

        def warn_once(message):
            if message not in logged_warnings:
                logged_warnings.add(message)
                logger.warning(message)

        # TODO param flag for CCS prediction
        def rolling_sum(a, n=2):
            ret = np.cumsum(a, axis=1, dtype=np.float32)
            ret[:, n:] = ret[:, n:] - ret[:, :-n]
            return ret[:, n - 1 :]

        ret_list = {}
        ret_list["matrix"] = {}
        ret_list["matrix_sum"] = {}
        ret_list["matrix_all"] = {}
        ret_list["pos_matrix"] = {}
        ret_list["matrix_hc"] = {}

        # Iterate over all instances
        for psm, row_index in zip(psm_list, indexes):
            peptidoform = psm.peptidoform
            seq = peptidoform.sequence
            seq_len = len(seq)
            charge = psm.get_precursor_charge()

            # For now anything longer than padding length is cut away
            # (C-terminal cutting)
            if seq_len > padding_length:
                seq = seq[0:padding_length]
                seq_len = len(seq)
                warn_once("Truncating peptide (too long): %s" % (seq))

            peptide_composition = [mass.std_aa_comp[aa] for aa in seq]

            # Initialize all feature matrices
            matrix = np.zeros(
                (padding_length, len(dict_index.keys())), dtype=np.float16
            )
            matrix_hc = np.zeros(
                (padding_length, len(dict_aa.keys())), dtype=np.float16
            )
            matrix_pos = np.zeros(
                (len(positions), len(dict_index.keys())), dtype=np.float16
            )

            for i, position_composition in enumerate(peptide_composition):
                for k, v in position_composition.items():
                    try:
                        matrix[i, dict_index[k]] = v
                    except IndexError:
                        warn_once(
                            f"Could not add the following value: pos {i} for atom {k} with value {v}"
                        )
                    except KeyError:
                        warn_once(
                            f"Could not add the following value: pos {i} for atom {k} with value {v}"
                        )

            for p in positions_pos:
                try:
                    aa = seq[p]
                except:
                    warn_once(f"Unable to get the following position: {p}")
                    continue
                for atom, val in mass.std_aa_comp[aa].items():
                    try:
                        matrix_pos[p, dict_index_pos[atom]] = val
                    except KeyError:
                        warn_once(f"Could not add the following atom: {atom}")
                    except IndexError:
                        warn_once(f"Could not add the following atom: {p} {atom} {val}")

            for pn in positions_neg:
                try:
                    aa = seq[seq_len + pn]
                except:
                    warn_once(f"Unable to get the following position: {p}")
                    continue
                for atom, val in mass.std_aa_comp[aa].items():
                    try:
                        matrix_pos[pn, dict_index_pos[atom]] = val
                    except KeyError:
                        warn_once(f"Could not add the following atom: {atom}")
                    except IndexError:
                        warn_once(
                            f"Could not add the following atom: {pn} {atom} {val}"
                        )

            for i, peptide_position in enumerate(peptidoform.parsed_sequence):
                try:
                    matrix_hc[i, dict_aa[peptide_position[0]]] = 1.0
                except KeyError:
                    warn_once(
                        f"Skipping the following (not in library): {i} {peptide_position}"
                    )
                except IndexError:
                    warn_once(
                        f"Could not add the following atom: {i} {peptide_position}"
                    )

                if peptide_position[1] is not None:
                    try:
                        modification_composition = peptide_position[1][0].composition
                    except KeyError:
                        warn_once(
                            f"Skipping the following (not in library): {peptide_position[1]}"
                        )
                        continue
                    except:
                        warn_once(
                            f"Skipping the following (not in library): {peptide_position[1]}"
                        )
                        continue

                    for (
                        atom_position_composition,
                        atom_change,
                    ) in modification_composition.items():
                        try:
                            matrix[
                                i, dict_index[atom_position_composition]
                            ] += atom_change
                            if i in positions:
                                matrix_pos[
                                    i, dict_index_pos[atom_position_composition]
                                ] += atom_change
                            elif i - seq_len in positions:
                                matrix_pos[
                                    i - seq_len,
                                    dict_index_pos[atom_position_composition],
                                ] += atom_change
                        except KeyError:
                            try:
                                warn_once(
                                    f"Could not add the following atom: {atom_position_composition}, attempting to replace the [] part"
                                )
                                atom_position_composition = sub(
                                    "\[.*?\]", "", atom_position_composition
                                )
                                matrix[
                                    i, dict_index[atom_position_composition]
                                ] += atom_change
                                if i in positions:
                                    matrix_pos[
                                        i, dict_index_pos[atom_position_composition]
                                    ] += atom_change
                                elif i - seq_len in positions:
                                    matrix_pos[
                                        i - seq_len,
                                        dict_index_pos[atom_position_composition],
                                    ] += atom_change
                            except KeyError:
                                warn_once(
                                    f"Could not add the following atom: {atom_position_composition}, second attempt, now ignored"
                                )
                                continue
                            except:
                                warn_once(
                                    f"Could not add the following atom: {atom_position_composition}, second attempt, now ignored"
                                )
                                continue
                        except IndexError:
                            warn_once(
                                f"Could not add the following atom: {i} {atom_position_composition} {atom_change}"
                            )
                        except:
                            warn_once(
                                f"Could not add the following atom: {atom_position_composition}, second attempt, now ignored"
                            )
                            continue

            matrix_all = np.sum(matrix, axis=0)
            matrix_all = np.append(matrix_all, seq_len)

            if predict_ccs:
                matrix_all = np.append(matrix_all, (seq.count("H")) / float(seq_len))
                matrix_all = np.append(
                    matrix_all,
                    (seq.count("F") + seq.count("W") + seq.count("Y")) / float(seq_len),
                )
                matrix_all = np.append(
                    matrix_all, (seq.count("D") + seq.count("E")) / float(seq_len)
                )
                matrix_all = np.append(
                    matrix_all, (seq.count("K") + seq.count("R")) / float(seq_len)
                )
                matrix_all = np.append(matrix_all, charge)

            matrix_sum = rolling_sum(matrix.T, n=2)[:, ::2].T

            ret_list["matrix"][row_index] = matrix
            ret_list["matrix_sum"][row_index] = matrix_sum
            ret_list["matrix_all"][row_index] = matrix_all
            ret_list["pos_matrix"][row_index] = matrix_pos.flatten()
            ret_list["matrix_hc"][row_index] = matrix_hc

        return ret_list

    def full_feat_extract(
        self,
        psm_list=[],
        seqs=[],
        mods=[],
        predict_ccs=False,
        identifiers=[],
        charges=[],
    ):
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
        charges : list
            optional list with charges, keep emtpy if these will not effect the predicted value

        Returns
        -------
        pd.DataFrame
            feature matrix
        """
        # TODO Reintroduce for CCS, check CCS flag
        if len(seqs) > 0:
            list_of_psms = []
            for seq, mod, id in zip(seqs, mods, identifiers):
                list_of_psms.append(
                    PSM(peptidoform=peprec_to_proforma(seq, mod), spectrum_id=id)
                )
            psm_list = PSMList(psm_list=list_of_psms)

        if self.verbose:
            t0 = time.time()

        # TODO pass CCS flag
        if self.add_sum_feat:
            if self.verbose:
                logger.debug("Extracting compositional sum features for modifications")
            X_feats_sum = self.get_feats_mods(psm_list, split_size=1, add_str="_sum")
        if self.ptm_add_feat:
            if self.verbose:
                logger.debug("Extracting compositional add features for modifications")
            X_feats_add = self.get_feats_mods(
                psm_list, split_size=self.split_size, add_str="_add"
            )
        if self.ptm_subtract_feat:
            if self.verbose:
                logger.debug(
                    "Extracting compositional subtract features for modifications"
                )
            X_feats_neg = self.get_feats_mods(
                psm_list,
                split_size=self.split_size,
                add_str="_subtract",
                subtract_mods=True,
            )
        if self.cnn_feats:
            if self.verbose:
                logger.debug("Extracting CNN features")
            X_cnn = self.encode_atoms(
                psm_list,
                list(range(len(psm_list))),
                charges=charges,
                predict_ccs=predict_ccs,
            )

        if self.cnn_feats:
            X = X_cnn
        if self.add_sum_feat:
            try:
                X = pd.concat([X, X_feats_sum], axis=1)
            except BaseException:
                X = X_feats_sum
        if self.ptm_add_feat:
            try:
                X = pd.concat([X, X_feats_add], axis=1)
            except BaseException:
                X = X_feats_add
        if self.ptm_subtract_feat:
            try:
                X = pd.concat([X, X_feats_neg], axis=1)
            except BaseException:
                X = X_feats_neg

        if self.verbose:
            logger.debug(
                "Time to calculate all features: %s seconds" % (time.time() - t0)
            )
        return X


def main(verbose=True):
    f_extractor = FeatExtractor(config_file="config.ini")
    df = pd.read_csv("parse_pride/seqs_exp.csv")
    df.index = ["Pep_" + str(dfi) for dfi in df.index]
    print(f_extractor.full_feat_extract(df["seq"], df["modifications"], df.index))


if __name__ == "__main__":
    main()
