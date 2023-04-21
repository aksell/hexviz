from enum import Enum

import streamlit as st
import torch
from tape import ProteinBertModel, TAPETokenizer
from tokenizers import Tokenizer
from transformers import (
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)


class ModelType(str, Enum):
    TAPE_BERT = "TapeBert"
    ZymCTRL = "ZymCTRL"
    PROT_BERT = "ProtBert"


class Model:
    def __init__(self, name, layers, heads):
        self.name: ModelType = name
        self.layers: int = layers
        self.heads: int = heads


@st.cache
def get_tape_bert() -> tuple[TAPETokenizer, ProteinBertModel]:
    tokenizer = TAPETokenizer()
    model = ProteinBertModel.from_pretrained("bert-base", output_attentions=True)
    return tokenizer, model


# Streamlit is not able to hash the tokenizer for ZymCTRL
# With streamlit 1.19 cache_object should work without this
@st.cache(hash_funcs={Tokenizer: lambda _: None})
def get_zymctrl() -> tuple[GPT2TokenizerFast, GPT2LMHeadModel]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("nferruz/ZymCTRL")
    model = GPT2LMHeadModel.from_pretrained("nferruz/ZymCTRL").to(device)
    return tokenizer, model


@st.cache(hash_funcs={BertTokenizer: lambda _: None})
def get_prot_bert() -> tuple[BertTokenizer, BertModel]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
    return tokenizer, model
