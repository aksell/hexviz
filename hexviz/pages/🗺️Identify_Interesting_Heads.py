import streamlit as st

from hexviz.attention import get_attention, get_sequence, get_structure
from hexviz.models import Model, ModelType
from hexviz.plot import plot_tiled_heatmap
from hexviz.view import (menu_items, select_heads_and_layers, select_model,
                         select_pdb, select_protein, select_sequence_slice)

st.set_page_config(layout="wide", menu_items=menu_items)
st.subheader("Find interesting heads and layers")


for k, v in st.session_state.items():
    st.session_state[k] = v

models = [
    Model(name=ModelType.TAPE_BERT, layers=12, heads=12),
    Model(name=ModelType.ZymCTRL, layers=36, heads=16),
]

with st.expander("Input a PDB id, upload a PDB file or input a sequence", expanded=True):
    pdb_id = select_pdb()
    uploaded_file = st.file_uploader("2.Upload PDB", type=["pdb"])
    input_sequence = st.text_area("3.Input sequence (Folded with ESMfold) Max 400 resis", "", key="input_sequence", max_chars=400)
    pdb_str, structure, source = select_protein(pdb_id, uploaded_file, input_sequence)
    st.write(f"Using: {source}")

selected_model = select_model(models)


structure = get_structure(pdb_id)

chains = list(structure.get_chains())
chain_ids = [chain.id for chain in chains]
if "selected_chain" not in st.session_state:
    st.session_state.selected_chain = chain_ids[0]
chain_selection = st.sidebar.selectbox(
    label = "Select Chain",
    options = chain_ids,
    key = "selected_chain",
)

selected_chain = next(chain for chain in chains if chain.id == chain_selection)
sequence = get_sequence(selected_chain)

l = len(sequence)
slice_start, slice_end = select_sequence_slice(l)
truncated_sequence = sequence[slice_start-1:slice_end]


layer_sequence, head_sequence = select_heads_and_layers(st.sidebar, selected_model)

st.markdown(f"Each tile is a heatmap of attention for a section of {pdb_id}({chain_selection}) from residue {slice_start} to {slice_end}. Adjust the section length and starting point in the sidebar.")

# TODO: Decide if you should get attention for the full sequence or just the truncated sequence
# Attention values will change depending on what we do.
attention = get_attention(sequence=truncated_sequence, model_type=selected_model.name)

fig = plot_tiled_heatmap(attention, layer_sequence=layer_sequence, head_sequence=head_sequence)
st.pyplot(fig)
