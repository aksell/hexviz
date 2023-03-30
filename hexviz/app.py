import py3Dmol
import stmol
import streamlit as st
from stmol import showmol

from hexviz.attention import get_attention_pairs
from hexviz.models import Model, ModelType

st.title("Attention Visualization on proteins")

"""
Visualize attention weights on protein structures for the protein language models ProtGPT2, TAPE-BERT and ZymCTRL.
Pick a PDB ID, layer and head to visualize attention.
"""

models = [
    # Model(name=ModelType.ProtGPT2, layers=36, heads=20),
    Model(name=ModelType.TAPE_BERT, layers=12, heads=12),
    Model(name=ModelType.ZymCTRL, layers=36, heads=16),
    # Model(name=ModelType.PROT_T5, layers=24, heads=32),
]

selected_model_name = st.selectbox("Select a model", [model.name.value for model in models], index=0)
selected_model = next((model for model in models if model.name.value == selected_model_name), None)

pdb_id = st.text_input("PDB ID", "4RW0")

left, right = st.columns(2)
with left:
    layer_one = st.number_input("Layer", value=1, min_value=1, max_value=selected_model.layers)
    layer = layer_one - 1
with right:
    head_one = st.number_input("Head", value=1, min_value=1, max_value=selected_model.heads)
    head = head_one - 1


with st.expander("Configure parameters", expanded=False):
    min_attn = st.slider("Minimum attention", min_value=0.0, max_value=0.4, value=0.1)
    try:
        structure = get_structure(pdb_id)
        ec_class = structure.header["compound"]["1"]["ec"]
    except KeyError:
        ec_class = None
    if ec_class and selected_model.name == ModelType.ZymCTRL:
        ec_class = st.text_input("Enzyme classification number fetched from PDB", ec_class)

attention_pairs = get_attention_pairs(pdb_id, layer, head, min_attn, model_type=selected_model.name)

def get_3dview(pdb):
    xyzview = py3Dmol.view(query=f"pdb:{pdb}")
    xyzview.setStyle({"cartoon": {"color": "spectrum"}})
    stmol.add_hover(xyzview, backgroundColor="black", fontColor="white")
    for att_weight, first, second in attention_pairs:
        stmol.add_cylinder(xyzview, start=first, end=second, cylradius=att_weight*3, cylColor='red', dashed=False)
    return xyzview


xyzview = get_3dview(pdb_id)
showmol(xyzview, height=500, width=800)
st.markdown(f'PDB: [{pdb_id}](https://www.rcsb.org/structure/{pdb_id})', unsafe_allow_html=True)

"""
More models will be added soon.
"""