from enum import Enum
from io import StringIO
from typing import List, Tuple
from urllib import request

import streamlit as st
import torch
from Bio.PDB import PDBParser, Polypeptide, Structure
from tape import ProteinBertModel, TAPETokenizer
from transformers import T5EncoderModel, T5Tokenizer


class ModelType(str, Enum):
    TAPE_BERT = "bert-base"
    PROT_T5 = "prot_t5_xl_half_uniref50-enc"


class Model:
    def __init__(self, name, layers, heads):
        self.name: ModelType = name
        self.layers: int = layers
        self.heads: int = heads


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

def get_sequences(structure: Structure) -> List[str]:
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

def get_protT5() -> Tuple[T5Tokenizer, T5EncoderModel]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(
        device
    )

    model.full() if device == "cpu" else model.half()

    return tokenizer, model

def get_tape_bert() -> Tuple[TAPETokenizer, ProteinBertModel]:
    tokenizer = TAPETokenizer()
    model = ProteinBertModel.from_pretrained('bert-base', output_attentions=True)
    return tokenizer, model

@st.cache_data
def get_attention(
    sequence: List[str], model_type: ModelType = ModelType.TAPE_BERT  
):
    if model_type == ModelType.TAPE_BERT:
        tokenizer, model = get_tape_bert()
        token_idxs = tokenizer.encode(sequence).tolist()
        inputs = torch.tensor(token_idxs).unsqueeze(0)
        with torch.no_grad():
            attns = model(inputs)[-1]
            # Remove attention from <CLS> (first) and <SEP> (last) token
        attns = [attn[:, :, 1:-1, 1:-1] for attn in attns]
        attns = torch.stack([attn.squeeze(0) for attn in attns])
    elif model_type == ModelType.PROT_T5:
            attns = None
            # Space separate sequences
            sequences = [" ".join(sequence) for sequence in sequences]
            tokenizer, model = get_protT5()
    else:
        raise ValueError(f"Model {model_type} not supported")

    return attns

def unidirectional_sum_filtered(attention, layer, head, threshold):
    num_layers, num_heads, seq_len, _ = attention.shape
    attention_head = attention[layer, head]
    unidirectional_sum_for_head = []
    for i in range(seq_len):
        for j in range(i, seq_len):
            # Attention matrices for BERT models are asymetric.
            # Bidirectional attention is reduced to one value by adding the
            # attention values
            # TODO think... does this operation make sense?
            sum = attention_head[i, j].item() + attention_head[j, i].item()
            if sum >= threshold:
                unidirectional_sum_for_head.append((sum, i, j))
    return unidirectional_sum_for_head

@st.cache_data
def get_attention_pairs(pdb_code: str, layer: int, head: int, threshold: int = 0.2, model_type: ModelType = ModelType.TAPE_BERT):
    # fetch structure
    structure = get_structure(pdb_code=pdb_code)
    # Get list of sequences
    sequences = get_sequences(structure)

    attention_pairs = []
    for i, sequence in enumerate(sequences):
        attention = get_attention(sequence=sequence, model_type=model_type)
        attention_unidirectional = unidirectional_sum_filtered(attention, layer, head, threshold)
        chain = list(structure.get_chains())[i]
        for attn_value, res_1, res_2 in attention_unidirectional:
            try:
                coord_1 = chain[res_1]["CA"].coord.tolist()
                coord_2 = chain[res_2]["CA"].coord.tolist()
            except KeyError:
                continue
            attention_pairs.append((attn_value, coord_1, coord_2))
        
    return attention_pairs
