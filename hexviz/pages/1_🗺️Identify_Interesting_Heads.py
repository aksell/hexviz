import re

import streamlit as st

from hexviz.attention import clean_and_validate_sequence, get_attention, res_to_1letter
from hexviz.config import URL
from hexviz.models import Model, ModelType
from hexviz.plot import plot_single_heatmap, plot_tiled_heatmap
from hexviz.view import (
    menu_items,
    select_heads_and_layers,
    select_model,
    select_pdb,
    select_protein,
    select_sequence_slice,
)

st.set_page_config(layout="wide", menu_items=menu_items)
st.title("Identify Interesting Heads")


for k, v in st.session_state.items():
    st.session_state[k] = v

models = [
    Model(name=ModelType.TAPE_BERT, layers=12, heads=12),
    Model(name=ModelType.ZymCTRL, layers=36, heads=16),
    Model(name=ModelType.PROT_BERT, layers=30, heads=16),
    Model(name=ModelType.PROT_T5, layers=24, heads=32),
]

with st.expander("Input a PDB id, upload a PDB file or input a sequence", expanded=True):
    pdb_id = select_pdb()
    uploaded_file = st.file_uploader("2.Upload PDB", type=["pdb"])
    input_sequence = st.text_area("3.Input sequence", "", key="input_sequence", max_chars=400)
    sequence, error = clean_and_validate_sequence(input_sequence)
    if error:
        st.error(error)
    pdb_str, structure, source = select_protein(pdb_id, uploaded_file, sequence)
    st.write(f"Visualizing: {source}")

selected_model = select_model(models)


chains = list(structure.get_chains())
chain_ids = [chain.id for chain in chains]
if "selected_chain" not in st.session_state:
    st.session_state.selected_chain = chain_ids[0]
chain_selection = st.sidebar.selectbox(
    label="Select Chain",
    options=chain_ids,
    key="selected_chain",
)

selected_chain = next(chain for chain in chains if chain.id == chain_selection)

ec_number = ""
if selected_model.name == ModelType.ZymCTRL:
    st.sidebar.markdown(
        """
        ZymCTRL EC number
        ---
        """
    )
    try:
        ec_number = structure.header["compound"]["1"]["ec"]
    except KeyError:
        pass
    ec_number = st.sidebar.text_input("Enzyme Comission number (EC)", ec_number)

    # Validate EC number
    if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ec_number):
        st.sidebar.error(
            """Please enter a valid Enzyme Commission number in the format of 4
            integers separated by periods (e.g., 1.2.3.21)"""
        )


residues = [res for res in selected_chain.get_residues()]
sequence = res_to_1letter(residues)

l = len(sequence)
slice_start, slice_end = select_sequence_slice(l)
truncated_sequence = sequence[slice_start - 1 : slice_end]
remove_special_tokens = st.sidebar.checkbox(
    "Hide attention to special tokens", key="remove_special_tokens"
)
if "fixed_scale" not in st.session_state:
    st.session_state.fixed_scale = True
fixed_scale = st.sidebar.checkbox("Fixed scale", help="For long sequences the default fixed 0 to 1 scale can have very low contrast heatmaps, consider using a relative scale to increase the contrast between high attention and low attention areas. Note that each subplot will have separate color scales so don't compare colors between attention heads if using a non-fixed scale.", key="fixed_scale")
if not fixed_scale:
    st.sidebar.warning("With `Fixed scale` set to False each cell in the grid has a dynamic color scale where the highest attention value in that cell is bright yellow. Colors can not be compared between cells.")


layer_sequence, head_sequence = select_heads_and_layers(st.sidebar, selected_model)

st.markdown(
    f"""Each tile is a heatmap of attention for a section of the {source} chain
    ({chain_selection}) from residue {slice_start} to {slice_end}. Adjust the
    section length and starting point in the sidebar."""
)

# TODO: Decide if you should get attention for the full sequence or just the truncated sequence
# Attention values will change depending on what we do.
attention, tokens = get_attention(
    sequence=truncated_sequence,
    model_type=selected_model.name,
    remove_special_tokens=remove_special_tokens,
    ec_number=ec_number,
)

fig = plot_tiled_heatmap(attention, layer_sequence=layer_sequence, head_sequence=head_sequence, fixed_scale=fixed_scale)


st.pyplot(fig)

st.subheader("Plot single head")

if selected_model.name == ModelType.PROT_T5:
    # Remove leading underscores from residue tokens
    tokens = [token[1:] if str(token) != "</s>" else token for token in tokens]

left, mid, right = st.columns(3)
with left:
    if "selected_layer" not in st.session_state:
        st.session_state["selected_layer"] = 5
    layer_one = st.selectbox(
        "Layer",
        options=[i for i in range(1, selected_model.layers + 1)],
        key="selected_layer",
    )
    layer = layer_one - 1
with mid:
    if "selected_head" not in st.session_state:
        st.session_state["selected_head"] = 1
    head_one = st.selectbox(
        "Head",
        options=[i for i in range(1, selected_model.heads + 1)],
        key="selected_head",
    )
    head = head_one - 1
with right:
    if "label_tokens" not in st.session_state:
        st.session_state.label_tokens = []
    tokens_to_label = st.multiselect("Label tokens", options=tokens, key="label_tokens")

if len(tokens_to_label) > 0:
    tokens = [token if token in tokens_to_label else "" for token in tokens]


single_head_fig = plot_single_heatmap(attention, layer, head, tokens=tokens, fixed_scale=fixed_scale)
st.pyplot(single_head_fig)
