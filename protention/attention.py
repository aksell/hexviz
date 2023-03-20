from io import StringIO
from urllib import request

import torch
from Bio.PDB import PDBParser, Structure
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

    # get model
    tokenizer, model = get_protT5()

    # call model

    # get attention

    # extract attention
