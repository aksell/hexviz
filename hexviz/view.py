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
    selected_model_name = st.selectbox("Select model", [model.name.value for model in models], index=get_selecte_model_index(models))
    st.session_state.selected_model_name = selected_model_name
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
    if pdb_id != stored_pdb:
        st.session_state.selected_chains = None
        st.session_state.selected_chain_index = 0
    st.session_state.pdb_id = pdb_id
    return pdb_id