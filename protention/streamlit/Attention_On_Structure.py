import py3Dmol
import stmol
import streamlit as st
from stmol import showmol

st.sidebar.title("pLM Attention Visualization")

st.title("pLM Attention Visualization")

pdb_id = st.text_input("PDB ID", "4RW0")
chain_id = None

left, right = st.columns(2)
with left:
    layer = st.number_input("Layer", value=8)
with right:
    head = st.number_input("Head", value=5)

min_attn = st.slider("Minimum attention", min_value=0.0, max_value=0.4, value=0.15)


def get_3dview(pdb):
    xyzview = py3Dmol.view(query=f"pdb:{pdb}")
    xyzview.setStyle({"cartoon": {"color": "spectrum"}})
    stmol.add_hover(xyzview, backgroundColor="black", fontColor="white")
    return xyzview


xyzview = get_3dview(pdb_id)
showmol(xyzview, height=500, width=800)
