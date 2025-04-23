import numpy as np
from psm_utils.psm import Peptidoform

from deeplc.feat_extractor import encode_peptidoform


def _check_result_structure(result: dict[str, np.ndarray]) -> None:
    expected_keys = {"matrix", "matrix_sum", "matrix_all", "pos_matrix", "matrix_hc"}
    assert expected_keys.issubset(result.keys())


def test_encode_peptidoform_without_modification():
    # Test using a simple peptidoform string without modifications.
    peptide = "ACDE/2"
    result = encode_peptidoform(peptide)
    _check_result_structure(result)

    # Default padding_length is 60 and there are 6 atoms in DEFAULT_DICT_INDEX.
    assert result["matrix"].shape == (60, 6)
    # matrix_all is formed by summing matrix (length 6) and appending the sequence length.
    # So without CCS prediction, matrix_all length should be 7.
    assert result["matrix_all"].shape == (7,)

    # The sequence is taken from the part before the '/'.
    expected_seq = "ACDE"
    # The last element of matrix_all should be the sequence length.
    assert result["matrix_all"][-1] == len(expected_seq)


def test_encode_peptidoform_with_modification():
    # Test using a peptidoform string with a modification.
    peptide = "AC[Carbamidomethyl]DE/2"
    result = encode_peptidoform(peptide)
    _check_result_structure(result)

    # Check basic structure as before.
    assert result["matrix"].shape == (60, 6)
    expected_seq = "ACDE"  # Modification does not alter the base sequence.
    assert result["matrix_all"][-1] == len(expected_seq)


def test_encode_peptidoform_with_ccs_prediction():
    # Test with predict_ccs=True.
    peptide = "ACDE/2"
    result = encode_peptidoform(peptide, predict_ccs=True)
    _check_result_structure(result)

    # Without predict_ccs, matrix_all has 7 elements (6 sums + sequence length).
    # With predict_ccs, 5 additional values are appended (ratios and charge),
    # resulting in a total length of 12.
    assert result["matrix_all"].shape == (12,)

    # Verify that the last element (charge) matches the precursor charge from Peptidoform.
    pf = Peptidoform(peptide)
    assert result["matrix_all"][-1] == pf.precursor_charge


def test_encode_peptidoform_feature_values() -> None:
    """
    Test that the returned feature values match the expected results
    for a known peptidoform.
    """
    peptide = "ACDE/2"
    result = encode_peptidoform(peptide, predict_ccs=False, padding_length=60)

    # For peptide "ACDE", using DEFAULT_DICT_INDEX = {"C":0,"H":1,"N":2,"O":3,"S":4,"P":5}
    # Based on pyteomics.mass.std_aa_comp, the expected composition is:
    # For 'A': { "C": 3, "H": 5, "N": 1, "O": 1 } -> row0: [3,5,1,1,0,0]
    # For 'C': { "C": 3, "H": 5, "N": 1, "O": 1, "S": 1 } -> row1: [3,5,1,1,1,0]
    # For 'D': { "C": 4, "H": 5, "N": 1, "O": 3 } -> row2: [4,5,1,3,0,0]
    # For 'E': { "C": 5, "H": 7, "N": 1, "O": 3 } -> row3: [5,7,1,3,0,0]
    #
    # Sum each column over positions 0 to 3:
    #   C: 3+3+4+5 = 15
    #   H: 5+5+5+7 = 22
    #   N: 1+1+1+1 = 4
    #   O: 1+1+3+3 = 8
    #   S: 0+1+0+0 = 1
    #   P: 0+0+0+0 = 0
    # Then sequence length appended: 4
    expected_matrix_all = np.array([15, 22, 4, 8, 1, 0, 4], dtype=np.float32)

    # Check matrix_all values using np.allclose.
    assert np.allclose(result["matrix_all"], expected_matrix_all), (
        f"Expected matrix_all: {expected_matrix_all}, got: {result['matrix_all']}"
    )

    # Also check the first row of the matrix for letter 'A'.
    # Expected for 'A': [3,5,1,1,0,0]
    expected_row_A = np.array([3, 5, 1, 1, 0, 0], dtype=np.float16)
    assert np.allclose(result["matrix"][0][:6], expected_row_A), (
        f"Expected row for A: {expected_row_A}, got: {result['matrix'][0][:6]}"
    )
