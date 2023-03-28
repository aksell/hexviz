from enum import Enum
from typing import Tuple

import streamlit as st
import torch
from tape import ProteinBertModel, TAPETokenizer
from transformers import (AutoTokenizer, GPT2LMHeadModel, T5EncoderModel,
                          T5Tokenizer)


class ModelType(str, Enum):
    TAPE_BERT = "TAPE-BERT"
    PROT_T5 = "prot_t5_xl_half_uniref50-enc"
    ZymCTRL = "ZymCTRL"
    ProtGPT2 = "ProtGPT2"


class Model:
    def __init__(self, name, layers, heads):
        self.name: ModelType = name
        self.layers: int = layers
        self.heads: int = heads

@st.cache
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

@st.cache
def get_tape_bert() -> Tuple[TAPETokenizer, ProteinBertModel]:
    tokenizer = TAPETokenizer()
    model = ProteinBertModel.from_pretrained('bert-base', output_attentions=True)
    return tokenizer, model

@st.cache
def get_zymctrl() -> Tuple[AutoTokenizer, GPT2LMHeadModel]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('nferruz/ZymCTRL')
    model = GPT2LMHeadModel.from_pretrained('nferruz/ZymCTRL').to(device)
    return tokenizer, model

@st.cache
def get_protgpt2() -> Tuple[AutoTokenizer, GPT2LMHeadModel]:
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2')
    model = GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2').to(device)
    return tokenizer, model
