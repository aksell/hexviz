from io import StringIO
from urllib import request

import torch
from Bio.PDB import PDBParser, Polypeptide, Structure
from transformers import T5EncoderModel, T5Tokenizer


def get_structure(pdb_code: str) -> Structure:
    """
    Get structure from PDB
    """
    pdb_url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
    pdb_data = request.urlopen(pdb_url).read().decode("utf-8")
    file = StringIO(pdb_data)
    parser = PDBParser()
    structure = parser.get_structure(pdb_code, file)
    return structure

def get_sequences(structure: Structure) -> list[str]:
    """
    Get list of sequences with residues on a single letter format

    Residues not in the standard 20 amino acids are replaced with X
    """
    sequences = []
    for seq in structure.get_chains():
        residues = [residue.get_resname() for residue in seq.get_residues()]
        # TODO ask if using protein_letters_3to1_extended makes sense
        residues_single_letter = map(lambda x: Polypeptide.protein_letters_3to1.get(x, "X"), residues)

        sequences.append(list(residues_single_letter))
    return sequences

def get_protT5() -> tuple[T5Tokenizer, T5EncoderModel]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(
        device
    )

    model.full() if device == "cpu" else model.half()

    return tokenizer, model


def get_attention(
    pdb_code: str, chain_ids: list[str], layer: int, head: int, min_attn: float = 0.2
):
    """
    Get attention from T5
    """
    # fetch structure
    structure = get_structure(pdb_code)
    # Get list of sequences
    sequences = get_sequences(structure)

    # get model
    tokenizer, model = get_protT5()

    # call model
    ## Get sequence

    # get attention

    # extract attention
