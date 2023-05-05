from io import StringIO
from urllib import request

import requests
import streamlit as st
import torch
from Bio.PDB import PDBParser, Polypeptide, Structure
from Bio.PDB.Residue import Residue

from hexviz.ec_number import ECNumber
from hexviz.models import ModelType, get_prot_bert, get_prot_t5, get_tape_bert, get_zymctrl


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


def res_to_1letter(residues: list[Residue]) -> str:
    """
    Get single letter sequence from a list or Residues

    Residues not in the standard 20 amino acids are replaced with X
    """
    res_names = [residue.get_resname() for residue in residues]
    residues_single_letter = map(lambda x: Polypeptide.protein_letters_3to1.get(x, "X"), res_names)

    return "".join(list(residues_single_letter))


def clean_and_validate_sequence(sequence: str) -> tuple[str, str | None]:
    lines = sequence.split("\n")
    cleaned_sequence = "".join(line.upper() for line in lines if not line.startswith(">"))
    cleaned_sequence = cleaned_sequence.replace(" ", "")
    valid_residues = set(Polypeptide.protein_letters_3to1.values())
    residues_in_sequence = set(cleaned_sequence)

    # Check if the sequence exceeds the max allowed length
    max_sequence_length = 400
    if len(cleaned_sequence) > max_sequence_length:
        error_message = (
            f"Sequence exceeds the max allowed length of {max_sequence_length} characters"
        )
        return cleaned_sequence, error_message

    illegal_residues = residues_in_sequence - valid_residues
    if illegal_residues:
        illegal_residues_str = ", ".join(illegal_residues)
        error_message = f"Sequence contains illegal residues: {illegal_residues_str}"
        return cleaned_sequence, error_message
    else:
        return cleaned_sequence, None


def remove_special_tokens_and_periods(attentions_tuple, sequence, tokenizer):
    tokens = tokenizer.tokenize(sequence)

    indices_to_remove = [
        i for i, token in enumerate(tokens) if token in {".", "<sep>", "<start>", "<end>", "<pad>"}
    ]

    new_attentions = []

    for attentions in attentions_tuple:
        # Remove rows and columns corresponding to special tokens and periods
        for idx in sorted(indices_to_remove, reverse=True):
            attentions = torch.cat((attentions[:, :, :idx], attentions[:, :, idx + 1 :]), dim=2)
            attentions = torch.cat(
                (attentions[:, :, :, :idx], attentions[:, :, :, idx + 1 :]), dim=3
            )

        # Append the modified attentions tensor to the new_attentions list
        new_attentions.append(attentions)

    return new_attentions, [token for i, token in enumerate(tokens) if i not in indices_to_remove]


@st.cache
def get_attention(
    sequence: str,
    model_type: ModelType = ModelType.TAPE_BERT,
    remove_special_tokens: bool = True,
    ec_number: str = None,
):
    """
    Returns a tensor of shape [n_layers, n_heads, n_res, n_res] with attention weights
    and the sequence of tokenes that the attention tensor corresponds to
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_type == ModelType.TAPE_BERT:
        tokenizer, model = get_tape_bert()
        token_idxs = tokenizer.encode(sequence).tolist()
        inputs = torch.tensor(token_idxs).unsqueeze(0)
        with torch.no_grad():
            attentions = model(inputs)[-1]
        if remove_special_tokens:
            # Remove attention from <CLS> (first) and <SEP> (last) token
            attentions = [attention[:, :, 1:-1, 1:-1] for attention in attentions]
            inputs = inputs[:, 1:-1]
        attentions = torch.stack([attention.squeeze(0) for attention in attentions])

    elif model_type == ModelType.ZymCTRL:
        tokenizer, model = get_zymctrl()

        if ec_number:
            sequence = f"{ec_number}<sep><start>{sequence}<end><pad>"

        inputs = tokenizer(sequence, return_tensors="pt").input_ids.to(device)
        attention_mask = tokenizer(sequence, return_tensors="pt").attention_mask.to(device)
        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask, output_attentions=True)
            attentions = outputs.attentions

        if ec_number and remove_special_tokens:
            # Remove attention to special tokens and periods separating EC number components
            attentions, inputs = remove_special_tokens_and_periods(attentions, sequence, tokenizer)

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

        if remove_special_tokens:
            # Remove attention from <CLS> (first) and <SEP> (last) token
            attentions = [attention[:, :, 1:-1, 1:-1] for attention in attentions]
            inputs = inputs[:, 1:-1]
        attentions = torch.stack([attention.squeeze(0) for attention in attentions])

    elif model_type == ModelType.PROT_T5:
        tokenizer, model = get_prot_t5()
        sequence_separated = " ".join(sequence)
        token_idxs = tokenizer.encode(sequence_separated)
        inputs = torch.tensor(token_idxs).unsqueeze(0).to(device)
        with torch.no_grad():
            attentions = model(inputs, output_attentions=True)[-1]

        if remove_special_tokens:
            # Remove attention to </s> (last) token
            attentions = [attention[:, :, :-1, :-1] for attention in attentions]
            inputs = inputs[:, :-1]
        attentions = torch.stack([attention.squeeze(0) for attention in attentions])

    else:
        raise ValueError(f"Model {model_type} not supported")

    input_ids_list = inputs.squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    # Transfer to CPU to avoid issues with streamlit caching
    return attentions.cpu(), tokens


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
    ec_numbers: list[list[ECNumber]] | None = None,
):
    """
    Note: All residue indexes returned are 0 indexed
    """
    structure = PDBParser().get_structure("pdb", StringIO(pdb_str))

    if chain_ids:
        chains = [ch for ch in structure.get_chains() if ch.id in chain_ids]
    else:
        chains = list(structure.get_chains())
    # Chains are treated at lists of residues to make indexing easier
    # and to avoid troubles with residues in PDB files not having a consistent
    # start index
    chain_ids = [chain.id for chain in chains]
    chains = [[res for res in chain.get_residues()] for chain in chains]

    attention_pairs = []
    top_residues = []

    ec_tag_length = 4

    def is_tag(x):
        return x < ec_tag_length

    for i, chain in enumerate(chains):
        ec_number = ec_numbers[i] if ec_numbers else None
        ec_string = ".".join([ec.number for ec in ec_number]) if ec_number else ""
        sequence = res_to_1letter(chain)
        attention, _ = get_attention(sequence=sequence, model_type=model_type, ec_number=ec_string)
        attention_unidirectional = unidirectional_avg_filtered(attention, layer, head, threshold)

        # Store sum of attention in to a resiue (from the unidirectional attention)
        residue_attention = {}
        for attn_value, res_1, res_2 in attention_unidirectional:
            try:
                if not ec_number:
                    coord_1 = chain[res_1]["CA"].coord.tolist()
                    coord_2 = chain[res_2]["CA"].coord.tolist()
                else:
                    if is_tag(res_1):
                        coord_1 = ec_number[res_1].coordinate
                    else:
                        coord_1 = chain[res_1 - ec_tag_length]["CA"].coord.tolist()
                    if is_tag(res_2):
                        coord_2 = ec_number[res_2].coordinate
                    else:
                        coord_2 = chain[res_2 - ec_tag_length]["CA"].coord.tolist()

            except KeyError:
                continue

            attention_pairs.append((attn_value, coord_1, coord_2))
            if not ec_number:
                residue_attention[res_1] = residue_attention.get(res_1, 0) + attn_value
                residue_attention[res_2] = residue_attention.get(res_2, 0) + attn_value
            else:
                for res in [res_1, res_2]:
                    if not is_tag(res):
                        residue_attention[res - ec_tag_length] = (
                            residue_attention.get(res - ec_tag_length, 0) + attn_value
                        )

        top_n_residues = sorted(residue_attention.items(), key=lambda x: x[1], reverse=True)[:top_n]

        for res, attn_sum in top_n_residues:
            coord = chain[res]["CA"].coord.tolist()
            top_residues.append((attn_sum, coord, chain_ids[i], res))

    return attention_pairs, top_residues
