from Bio.PDB.Structure import Structure

from protention.attention import get_structure


def test_get_structure():
    pdb_id = "1AKE"
    structure = get_structure(pdb_id)
    assert structure is not None
    assert isinstance(structure, Structure)
