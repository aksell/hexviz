import streamlit as st

menu_items = {
    "Get Help": "https://huggingface.co/spaces/aksell/hexviz/discussions/new", 
    "Report a bug": "https://huggingface.co/spaces/aksell/hexviz/discussions/new", 
    "About": "Created by [Aksel Lenes](https://github.com/aksell/) from Noelia Ferruz's group at the Institute of Molecular Biology of Barcelona. Read more at https://www.aiproteindesign.com/"
    }

def get_selecte_model_index(models):
    selected_model_name = st.session_state.get("selected_model_name", None)
    if selected_model_name is None:
        return 0
    else:
        return next((i for i, model in enumerate(models) if model.name.value == selected_model_name), None)

def select_model(models):
    """
    Select model, prefil selector with selected model from session storage

    Saves the selected model in session storage.
    """
    stored_model  = st.session_state.get("selected_model_name", None)
    selected_model_name = st.selectbox("Select model", [model.name.value for model in models], index=get_selecte_model_index(models))
    st.session_state.selected_model_name = selected_model_name
    model_changed = stored_model != selected_model_name
    if model_changed:
        if "plot_heads" in st.session_state:
            del st.session_state.plot_heads
        if "plot_layers" in st.session_state:
            del st.session_state.plot_layers
    select_model = next((model for model in models if model.name.value == selected_model_name), None)
    return select_model

def select_pdb():
    st.sidebar.markdown(
        """
        Select Protein 
        ---
        """)
    stored_pdb = st.session_state.get("pdb_id", None)
    pdb_id = st.sidebar.text_input(
            label="PDB ID",
            value=stored_pdb or "2FZ5")
    pdb_changed = stored_pdb != pdb_id
    if pdb_changed:
        st.session_state.selected_chains = None
        st.session_state.selected_chain_index = 0
        if "sequence_slice" in st.session_state:
            del st.session_state.sequence_slice
    st.session_state.pdb_id = pdb_id
    return pdb_id

def select_heads_and_layers(sidebar, model):
    sidebar.markdown(
        """
        Select Heads and Layers
        ---
        """
    )
    if "plot_heads" not in st.session_state:
        "setting heads"
        st.session_state.plot_heads = (1, model.heads//2)
    if "plot_layers" not in st.session_state:
        "setting layers"
        st.session_state.plot_layers = (1, model.layers//2)
    head_range = sidebar.slider("Heads to plot", min_value=1, max_value=model.heads, value=st.session_state.plot_heads, step=1)
    st.session_state.plot_heads = head_range
    layer_range = sidebar.slider("Layers to plot", min_value=1, max_value=model.layers, value=st.session_state.plot_layers, step=1)
    st.session_state.plot_layers = layer_range

    step_size = sidebar.number_input("Optional step size to skip heads and layers", value=1, min_value=1, max_value=model.layers)
    layer_sequence = list(range(layer_range[0]-1, layer_range[1], step_size))
    head_sequence = list(range(head_range[0]-1, head_range[1], step_size))

    return layer_sequence, head_sequence

def select_sequence_slice(sequence_length):
    st.sidebar.markdown("Sequence segment to plot")
    if "sequence_slice" not in st.session_state:
        st.session_state["sequence_slice"] = (1, min(50, sequence_length))
    slice = st.sidebar.slider("Sequence", value=st.session_state.sequence_slice, min_value=1, max_value=sequence_length, step=1)
    st.session_state.sequence_slice = slice
    return slice