from io import StringIO
from urllib import request

import requests
import streamlit as st
import torch
from Bio.PDB import PDBParser, Polypeptide, Structure

from hexviz.models import (
    ModelType,
    get_prot_bert,
    get_prot_t5,
    get_tape_bert,
    get_zymctrl,
)


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


def get_pdb_file(pdb_code: str) -> Structure:
    """
    Get structure from PDB
    """
    pdb_url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
    pdb_data = request.urlopen(pdb_url).read().decode("utf-8")
    file = StringIO(pdb_data)
    return file


@st.cache
def get_pdb_from_seq(sequence: str) -> str:
    """
    Get structure from sequence
    """
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    res = requests.post(url, data=sequence)
    pdb_str = res.text
    return pdb_str


def get_chains(structure: Structure) -> list[str]:
    """
    Get list of chains in a structure
    """
    chains = []
    for model in structure:
        for chain in model.get_chains():
            chains.append(chain.id)
    return chains


def get_sequence(chain) -> str:
    """
    Get sequence from a chain

    Residues not in the standard 20 amino acids are replaced with X
    """
    residues = [residue.get_resname() for residue in chain.get_residues()]
    # TODO ask if using protein_letters_3to1_extended makes sense
    residues_single_letter = map(
        lambda x: Polypeptide.protein_letters_3to1.get(x, "X"), residues
    )

    return "".join(list(residues_single_letter))


def clean_and_validate_sequence(sequence: str) -> tuple[str, str | None]:
    lines = sequence.split("\n")
    cleaned_sequence = "".join(
        line.upper() for line in lines if not line.startswith(">")
    )
    cleaned_sequence = cleaned_sequence.replace(" ", "")
    valid_residues = set(Polypeptide.protein_letters_3to1.values())
    residues_in_sequence = set(cleaned_sequence)

    # Check if the sequence exceeds the max allowed length
    max_sequence_length = 400
    if len(cleaned_sequence) > max_sequence_length:
        error_message = f"Sequence exceeds the max allowed length of {max_sequence_length} characters"
        return cleaned_sequence, error_message

    illegal_residues = residues_in_sequence - valid_residues
    if illegal_residues:
        illegal_residues_str = ", ".join(illegal_residues)
        error_message = f"Sequence contains illegal residues: {illegal_residues_str}"
        return cleaned_sequence, error_message
    else:
        return cleaned_sequence, None


@st.cache
def get_attention(sequence: str, model_type: ModelType = ModelType.TAPE_BERT):
    """
    Returns a tensor of shape [n_layers, n_heads, n_res, n_res] with attention weights
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_type == ModelType.TAPE_BERT:
        tokenizer, model = get_tape_bert()
        token_idxs = tokenizer.encode(sequence).tolist()
        inputs = torch.tensor(token_idxs).unsqueeze(0)
        with torch.no_grad():
            attentions = model(inputs)[-1]
            # Remove attention from <CLS> (first) and <SEP> (last) token
        attentions = [attention[:, :, 1:-1, 1:-1] for attention in attentions]
        attentions = torch.stack([attention.squeeze(0) for attention in attentions])

    elif model_type == ModelType.ZymCTRL:
        tokenizer, model = get_zymctrl()
        inputs = tokenizer(sequence, return_tensors="pt").input_ids.to(device)
        attention_mask = tokenizer(sequence, return_tensors="pt").attention_mask.to(
            device
        )

        with torch.no_grad():
            outputs = model(
                inputs, attention_mask=attention_mask, output_attentions=True
            )
            attentions = outputs.attentions

        # torch.Size([1, n_heads, n_res, n_res]) -> torch.Size([n_heads, n_res, n_res])
        attention_squeezed = [torch.squeeze(attention) for attention in attentions]
        # ([n_heads, n_res, n_res]*n_layers) -> [n_layers, n_heads, n_res, n_res]
        attention_stacked = torch.stack([attention for attention in attention_squeezed])
        attentions = attention_stacked

    elif model_type == ModelType.PROT_BERT:
        tokenizer, model = get_prot_bert()
        sequence_separated = " ".join(sequence)
        token_idxs = tokenizer.encode(sequence_separated)
        inputs = torch.tensor(token_idxs).unsqueeze(0).to(device)
        with torch.no_grad():
            attentions = model(inputs, output_attentions=True)[-1]
            # Remove attention from <CLS> (first) and <SEP> (last) token
        attentions = [attention[:, :, 1:-1, 1:-1] for attention in attentions]
        attentions = torch.stack([attention.squeeze(0) for attention in attentions])

    elif model_type == ModelType.PROT_T5:
        tokenizer, model = get_prot_t5()
        sequence_separated = " ".join(sequence)
        token_idxs = tokenizer.encode(sequence_separated)
        inputs = torch.tensor(token_idxs).unsqueeze(0).to(device)
        with torch.no_grad():
            attentions = model(inputs, output_attentions=True)[
                -1
            ]  # Do you need an attention mask?

        # Remove attention to <pad> (first) and <extra_id_1>, <extra_id_2> (last) tokens
        attentions = [attention[:, :, 3:-3, 3:-3] for attention in attentions]
        attentions = torch.stack([attention.squeeze(0) for attention in attentions])

    else:
        raise ValueError(f"Model {model_type} not supported")

    # Transfer to CPU to avoid issues with streamlit caching
    return attentions.cpu()


def unidirectional_avg_filtered(attention, layer, head, threshold):
    num_layers, num_heads, seq_len, _ = attention.shape
    attention_head = attention[layer, head]
    unidirectional_avg_for_head = []
    for i in range(seq_len):
        for j in range(i, seq_len):
            # Attention matrices for BERT models are asymetric.
            # Bidirectional attention is represented by the average of the two values
            sum = attention_head[i, j].item() + attention_head[j, i].item()
            avg = sum / 2
            if avg >= threshold:
                unidirectional_avg_for_head.append((avg, i, j))
    return unidirectional_avg_for_head


# Passing the pdb_str here is a workaround for streamlit caching
# where I need the input to be hashable and not changing
# The ideal would be to pass in the structure directly, not parsing
# Thist twice. If streamlit is upgaded to past 0.17 this can be
# fixed.
@st.cache
def get_attention_pairs(
    pdb_str: str,
    layer: int,
    head: int,
    chain_ids: list[str] | None,
    threshold: int = 0.2,
    model_type: ModelType = ModelType.TAPE_BERT,
    top_n: int = 2,
):
    structure = PDBParser().get_structure("pdb", StringIO(pdb_str))
    if chain_ids:
        chains = [ch for ch in structure.get_chains() if ch.id in chain_ids]
    else:
        chains = list(structure.get_chains())

    attention_pairs = []
    top_residues = []
    for chain in chains:
        sequence = get_sequence(chain)
        attention = get_attention(sequence=sequence, model_type=model_type)
        attention_unidirectional = unidirectional_avg_filtered(
            attention, layer, head, threshold
        )

        # Store sum of attention in to a resiue (from the unidirectional attention)
        residue_attention = {}
        for attn_value, res_1, res_2 in attention_unidirectional:
            try:
                coord_1 = chain[res_1]["CA"].coord.tolist()
                coord_2 = chain[res_2]["CA"].coord.tolist()
            except KeyError:
                continue

            attention_pairs.append(
                (attn_value, coord_1, coord_2, chain.id, res_1, res_2)
            )
            residue_attention[res_1] = residue_attention.get(res_1, 0) + attn_value
            residue_attention[res_2] = residue_attention.get(res_2, 0) + attn_value

        top_n_residues = sorted(
            residue_attention.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        for res, attn_sum in top_n_residues:
            coord = chain[res]["CA"].coord.tolist()
            top_residues.append((attn_sum, coord, chain.id, res))

    return attention_pairs, top_residues
