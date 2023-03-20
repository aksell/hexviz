from io import StringIO
from urllib import request

from Bio.PDB import PDBParser, Structure


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


def get_attention(
    pdb_code: str, chain_ids: list[str], layer: int, head: int, min_attn: float = 0.2
):
    """
    Get attention from T5
    """
    # fetch structure
    structure = get_structure(pdb_code)

    # get model

    # call model

    # get attention

    # extract attention
