import streamlit as st

from hexviz.attention import get_attention, get_sequence, get_structure
from hexviz.models import Model, ModelType
from hexviz.plot import plot_tiled_heatmap
from hexviz.view import get_selecte_model_index

st.set_page_config(layout="wide")
st.subheader("Find interesting heads and layers")


models = [
    Model(name=ModelType.TAPE_BERT, layers=12, heads=12),
    Model(name=ModelType.ZymCTRL, layers=36, heads=16),
]

selected_model_name = st.selectbox("Select a model", [model.name.value for model in models], index=get_selecte_model_index(models))
st.session_state.selected_model_name = selected_model_name
selected_model = next((model for model in models if model.name.value == selected_model_name), None)

pdb_id = st.sidebar.text_input(
        label="PDB ID",
        value=st.session_state.get("pdb_id", "2FZ5"),
    )
st.session_state.pdb_id = pdb_id


structure = get_structure(pdb_id)

chains = list(structure.get_chains())
chain_ids = [chain.id for chain in chains]
chain_selection = st.sidebar.selectbox(
    label="Select Chain",
    options=chain_ids,
    index=st.session_state.get("selected_chain_index", 0)
)
st.session_state.selected_chain_index = chain_ids.index(chain_selection)

selected_chain = next(chain for chain in chains if chain.id == chain_selection)
sequence = get_sequence(selected_chain)

l = len(sequence)
st.sidebar.markdown("Sequence segment to plot")
slice_start, slice_end = st.sidebar.slider("Sequence", min_value=1, max_value=l, value=(1, 50), step=1)
# slice_start= st.sidebar.number_input(f"Section start(1-{l})",value=1, min_value=1, max_value=l)
# slice_end = st.sidebar.number_input(f"Section end(1-{l})",value=50, min_value=1, max_value=l)
truncated_sequence = sequence[slice_start-1:slice_end]

head_range = st.sidebar.slider("Heads to plot", min_value=1, max_value=selected_model.heads, value=(1, selected_model.heads//2), step=1)
layer_range = st.sidebar.slider("Layers to plot", min_value=1, max_value=selected_model.layers, value=(1, selected_model.layers//2), step=1)
step_size = st.sidebar.number_input("Optional step size to skip heads and layers", value=1, min_value=1, max_value=selected_model.layers)
layer_sequence = list(range(layer_range[0]-1, layer_range[1], step_size))
head_sequence = list(range(head_range[0]-1, head_range[1], step_size))


st.markdown(f"Each tile is a heatmap of attention for a section of {pdb_id}({chain_selection}) from residue {slice_start} to {slice_end}. Adjust the section length and starting point in the sidebar.")

# TODO: Decide if you should get attention for the full sequence or just the truncated sequence
# Attention values will change depending on what we do.
attention = get_attention(sequence=truncated_sequence, model_type=selected_model.name)

fig = plot_tiled_heatmap(attention, layer_sequence=layer_sequence, head_sequence=head_sequence)
st.pyplot(fig)
