from Bio.PDB.Structure import Structure
from transformers import T5EncoderModel, T5Tokenizer

from protention.attention import get_protT5, get_sequences, get_structure


def test_get_structure():
    pdb_id = "1AKE"
    structure = get_structure(pdb_id)

    assert structure is not None
    assert isinstance(structure, Structure)

def test_get_sequences():
    pdb_id = "1AKE"
    structure = get_structure(pdb_id)
    
    sequences = get_sequences(structure)

    assert sequences is not None
    assert len(sequences) == 2

    A, B = sequences
    assert A[:3] == ["M", "R", "I"]

def test_get_protT5():
    result = get_protT5()

    assert result is not None
    assert isinstance(result, tuple)

    tokenizer, model = result

    assert isinstance(tokenizer, T5Tokenizer)
    assert isinstance(model, T5EncoderModel)
