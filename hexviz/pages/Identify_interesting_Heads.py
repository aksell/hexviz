import streamlit as st

from hexviz.attention import get_attention, get_sequence, get_structure
from hexviz.models import Model, ModelType
from hexviz.plot import plot_tiled_heatmap

st.set_page_config(layout="wide")


models = [
    Model(name=ModelType.TAPE_BERT, layers=12, heads=12),
    Model(name=ModelType.ZymCTRL, layers=36, heads=16),
]

selected_model_name = st.sidebar.selectbox("Select a model", [model.name.value for model in models], index=0)
selected_model = next((model for model in models if model.name.value == selected_model_name), None)

pdb_id = st.sidebar.text_input(
        label="PDB ID",
        value="1AKE",
    )

structure = get_structure(pdb_id)
chains = list(structure.get_chains())

sequence = get_sequence(chains[0])
l = len(sequence)
n_residues = st.sidebar.number_input(f"Residue count (1-{l})",value=50, min_value=1, max_value=l)
truncated_sequence = sequence[:n_residues]

attention = get_attention(sequence=truncated_sequence, model_type=selected_model.name)

st.subheader("Find interesting heads and layers")
fig = plot_tiled_heatmap(attention, layer_count=selected_model.layers, head_count=selected_model.heads)
st.pyplot(fig)
