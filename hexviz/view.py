import streamlit as st


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
    pdb_id = st.sidebar.text_input(
            label="PDB ID",
            value=st.session_state.get("pdb_id", "2FZ5"))
    st.session_state.pdb_id = pdb_id
    return pdb_id