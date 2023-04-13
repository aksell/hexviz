from enum import Enum

import streamlit as st
import torch
from tape import ProteinBertModel, TAPETokenizer
from transformers import (AutoTokenizer, BertModel, BertTokenizer,
                          GPT2LMHeadModel, GPT2TokenizerFast)


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
def get_tape_bert() -> tuple[TAPETokenizer, ProteinBertModel]:
    tokenizer = TAPETokenizer()
    model = ProteinBertModel.from_pretrained('bert-base', output_attentions=True)
    return tokenizer, model

@st.cache
def get_zymctrl() -> tuple[GPT2TokenizerFast, GPT2LMHeadModel]:
    device = torch.device("cuda:0" if False and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('nferruz/ZymCTRL')
    model = GPT2LMHeadModel.from_pretrained('nferruz/ZymCTRL').to(device)
    # here we return things that are on the GPU, streamlit can't cache that I think like it can't cache
    # the attention weights. Figure out how to do caching of GPU stuff in streamlit better.
    # maybe object caching + streamlit 1.19?
    return tokenizer, model

@st.cache
def get_prot_bert() -> tuple[BertTokenizer, BertModel]:
    # TODO use cuda device here too?
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    return tokenizer, model
