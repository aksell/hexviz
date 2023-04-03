import streamlit as st


def get_selecte_model_index(models):
    selected_model_name = st.session_state.get("selected_model_name", None)
    if selected_model_name is None:
        return 0
    else:
        return next((i for i, model in enumerate(models) if model.name.value == selected_model_name), None)