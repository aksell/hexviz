import py3Dmol
import stmol
import streamlit as st
from stmol import showmol

from protention.attention import Model, ModelType, get_attention

st.sidebar.title("pLM Attention Visualization")

st.title("pLM Attention Visualization")

# Define list of model types
models = [
    Model(name=ModelType.TAPE_BERT, layers=12, heads=12),
]

selected_model_name = st.selectbox("Select a model", [model.name.value for model in models], index=0)
selected_model = next((model for model in models if model.name.value == selected_model_name), None)

pdb_id = st.text_input("PDB ID", "4RW0")

left, right = st.columns(2)
with left:
    layer = st.number_input("Layer", value=1, min_value=1, max_value=selected_model.layers)
with right:
    head = st.number_input("Head", value=1, min_value=1, max_value=selected_model.heads)

min_attn = st.slider("Minimum attention", min_value=0.0, max_value=0.4, value=0.15)

attention = get_attention(pdb_id, model=selected_model.name)

def get_3dview(pdb):
    xyzview = py3Dmol.view(query=f"pdb:{pdb}")
    xyzview.setStyle({"cartoon": {"color": "spectrum"}})
    stmol.add_hover(xyzview, backgroundColor="black", fontColor="white")
    return xyzview


xyzview = get_3dview(pdb_id)
showmol(xyzview, height=500, width=800)
