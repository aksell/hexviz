from io import StringIO

import streamlit as st
from Bio.PDB import PDBParser

from hexviz.attention import get_pdb_file, get_pdb_from_seq

menu_items = {
    "Get Help": "https://huggingface.co/spaces/aksell/hexviz/discussions/new",
    "Report a bug": "https://huggingface.co/spaces/aksell/hexviz/discussions/new",
    "About": "Created by [Aksel Lenes](https://github.com/aksell/) from Noelia Ferruz's group at the Institute of Molecular Biology of Barcelona. Read more at https://www.aiproteindesign.com/",
}


def get_selecte_model_index(models):
    selected_model_name = st.session_state.get("selected_model_name", None)
    if selected_model_name is None:
        return 0
    else:
        return next(
            (i for i, model in enumerate(models) if model.name.value == selected_model_name),
            None,
        )


def clear_model_state():
    if "plot_heads" in st.session_state:
        del st.session_state.plot_heads
    if "plot_layers" in st.session_state:
        del st.session_state.plot_layers
    if "selected_head" in st.session_state:
        del st.session_state.selected_head
    if "selected_layer" in st.session_state:
        del st.session_state.selected_layer
    if "plot_layers" in st.session_state:
        del st.session_state.plot_layers
    if "plot_heads" in st.session_state:
        del st.session_state.plot_heads
    if "label_tokens" in st.session_state:
        del st.session_state.label_tokens


def select_model(models):
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = models[0].name.value
    selected_model_name = st.selectbox(
        "Select model",
        [model.name.value for model in models],
        key="selected_model_name",
        on_change=clear_model_state,
    )
    select_model = next(
        (model for model in models if model.name.value == selected_model_name), None
    )
    return select_model


def clear_pdb_state():
    if "selected_chains" in st.session_state:
        del st.session_state.selected_chains
    if "selected_chain" in st.session_state:
        del st.session_state.selected_chain
    if "sequence_slice" in st.session_state:
        del st.session_state.sequence_slice
    if "uploaded_pdb_str" in st.session_state:
        del st.session_state.uploaded_pdb_str


def select_pdb():
    if "pdb_id" not in st.session_state:
        st.session_state.pdb_id = "2FZ5"
    pdb_id = st.text_input(label="1.PDB ID", key="pdb_id", on_change=clear_pdb_state)
    return pdb_id


def select_protein(pdb_code, uploaded_file, input_sequence):
    # We get the pdb from 1 of 3 places:
    # 1. Cached pdb from session storage
    # 2. PDB file from uploaded file
    # 3. PDB file fetched based on the pdb_code input
    parser = PDBParser()
    if uploaded_file is not None:
        pdb_str = uploaded_file.read().decode("utf-8")
        st.session_state["uploaded_pdb_str"] = pdb_str
        source = f"uploaded pdb file {uploaded_file.name}"
        structure = parser.get_structure("Userfile", StringIO(pdb_str))
    elif input_sequence:
        pdb_str = get_pdb_from_seq(str(input_sequence))
        if not pdb_str:
            st.error("ESMfold error, unable to fold sequence")
            return None, None, None
        else:
            structure = parser.get_structure("ESMFold", StringIO(pdb_str))
            if "selected_chains" in st.session_state:
                del st.session_state.selected_chains
            source = "Input sequence + ESM-fold"
    elif "uploaded_pdb_str" in st.session_state:
        pdb_str = st.session_state.uploaded_pdb_str
        source = "Uploaded file stored in cache"
        structure = parser.get_structure("userfile", StringIO(pdb_str))
    else:
        file = get_pdb_file(pdb_code)
        pdb_str = file.read()
        source = f"PDB ID: {pdb_code}"
        structure = parser.get_structure(pdb_code, StringIO(pdb_str))

    return pdb_str, structure, source


def select_heads_and_layers(sidebar, model):
    sidebar.markdown(
        """
        Select Heads and Layers
        ---
        """
    )
    if "plot_heads" not in st.session_state:
        st.session_state.plot_heads = (1, model.heads // 2)
    head_range = sidebar.slider(
        "Heads to plot", min_value=1, max_value=model.heads, key="plot_heads", step=1
    )
    if "plot_layers" not in st.session_state:
        st.session_state.plot_layers = (1, model.layers // 2)
    layer_range = sidebar.slider(
        "Layers to plot", min_value=1, max_value=model.layers, key="plot_layers", step=1
    )

    if "plot_step_size" not in st.session_state:
        st.session_state.plot_step_size = 1
    step_size = sidebar.number_input(
        "Optional step size to skip heads and layers",
        key="plot_step_size",
        min_value=1,
        max_value=model.layers,
    )
    layer_sequence = list(range(layer_range[0] - 1, layer_range[1], step_size))
    head_sequence = list(range(head_range[0] - 1, head_range[1], step_size))

    return layer_sequence, head_sequence


def select_sequence_slice(sequence_length):
    st.sidebar.markdown(
        """
        Sequence segment to plot
        ---
        """
    )
    if "sequence_slice" not in st.session_state:
        st.session_state.sequence_slice = (1, min(50, sequence_length))
    slice = st.sidebar.slider(
        "Sequence", key="sequence_slice", min_value=1, max_value=sequence_length, step=1
    )
    return slice
