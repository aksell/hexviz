from enum import Enum
from typing import Tuple

import streamlit as st
import torch
from tape import ProteinBertModel, TAPETokenizer
from transformers import (AutoTokenizer, BertForMaskedLM, BertTokenizer,
                          GPT2LMHeadModel)


class ModelType(str, Enum):
    TAPE_BERT = "TAPE-BERT"
    ZymCTRL = "ZymCTRL"
    PROT_BERT = "ProtBert"


class Model:
    def __init__(self, name, layers, heads):
        self.name: ModelType = name
        self.layers: int = layers
        self.heads: int = heads


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
def get_prot_bert() -> Tuple[BertTokenizer, BertForMaskedLM]:
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
    return tokenizer, model
