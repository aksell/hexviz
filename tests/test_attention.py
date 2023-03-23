import torch
from Bio.PDB.Structure import Structure
from transformers import T5EncoderModel, T5Tokenizer

from hexviz.attention import (ModelType, get_attention, get_protT5,
                              get_sequences, get_structure,
                              unidirectional_sum_filtered)


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

def test_get_attention_tape():

    result = get_attention("1AKE", model=ModelType.TAPE_BERT)

    assert result is not None
    assert result.shape == torch.Size([12,12,456,456])

def test_get_unidirection_sum_filtered():
    # 1 head, 1 layer, 4 residues long attention tensor
    attention= torch.tensor([[[[1, 2, 3, 4],
                               [2, 5, 6, 7],
                               [3, 6, 8, 9],
                               [4, 7, 9, 11]]]], dtype=torch.float32)

    result = unidirectional_sum_filtered(attention, 0, 0, 0)

    assert result is not None
    assert len(result) == 10

    attention= torch.tensor([[[[1, 2, 3],
                               [2, 5, 6],
                               [4, 7, 91]]]], dtype=torch.float32)

    result = unidirectional_sum_filtered(attention, 0, 0, 0)

    assert len(result) == 6
