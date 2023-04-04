import pandas as pd
import py3Dmol
import stmol
import streamlit as st
from stmol import showmol

from hexviz.attention import get_attention_pairs, get_chains, get_structure
from hexviz.models import Model, ModelType
from hexviz.view import menu_items, select_model, select_pdb

st.set_page_config(layout="centered", menu_items=menu_items)
st.title("Attention Visualization on proteins")


models = [
    Model(name=ModelType.TAPE_BERT, layers=12, heads=12),
    Model(name=ModelType.ZymCTRL, layers=36, heads=16),
]

pdb_id = select_pdb()
structure = get_structure(pdb_id)
chains = get_chains(structure)

selected_chains = st.sidebar.multiselect(label="Select Chain(s)", options=chains, default=st.session_state.get("selected_chains", None) or chains)
st.session_state.selected_chains = selected_chains


st.sidebar.markdown(
    """
    Attention parameters
    ---
    """)
min_attn = st.sidebar.slider("Minimum attention", min_value=0.0, max_value=0.4, value=0.1)
n_highest_resis = st.sidebar.number_input("Num highest attention resis to label", value=2, min_value=1, max_value=100)
label_highest = st.sidebar.checkbox("Label highest attention residues", value=True)
sidechain_highest = st.sidebar.checkbox("Show sidechains", value=True)
# TODO add avg or max attention as params


with st.sidebar.expander("Label residues manually"):
    hl_chain = st.selectbox(label="Chain to label", options=selected_chains, index=0)
    hl_resi_list = st.multiselect(label="Selected Residues",options=list(range(1,5000)))

    label_resi = st.checkbox(label="Label Residues", value=True)


left, mid, right = st.columns(3)
with left:
    selected_model = select_model(models)
with mid:
    layer_one = st.number_input("Layer",value=st.session_state.get("selected_layer", 5), min_value=1, max_value=selected_model.layers)
    st.session_state["selected_layer"] = layer_one
    layer = layer_one - 1
with right:
    head_one = st.number_input("Head", value=st.session_state.get("selected_head", 1), min_value=1, max_value=selected_model.heads)
    st.session_state["selected_head"] = head_one
    head = head_one - 1


if selected_model.name == ModelType.ZymCTRL:
    try:
        ec_class = structure.header["compound"]["1"]["ec"]
    except KeyError:
        ec_class = None
    if ec_class and selected_model.name == ModelType.ZymCTRL:
        ec_class = st.sidebar.text_input("Enzyme classification number fetched from PDB", ec_class)

attention_pairs, top_residues = get_attention_pairs(pdb_id, chain_ids=selected_chains, layer=layer, head=head, threshold=min_attn, model_type=selected_model.name, top_n=n_highest_resis)

sorted_by_attention = sorted(attention_pairs, key=lambda x: x[0], reverse=True) 

def get_3dview(pdb):
    xyzview = py3Dmol.view(query=f"pdb:{pdb}")
    xyzview.setStyle({"cartoon": {"color": "spectrum"}})
    stmol.add_hover(xyzview, backgroundColor="black", fontColor="white")


    hidden_chains = [x for x in chains if x not in selected_chains]
    for chain in hidden_chains:
        xyzview.setStyle({"chain": chain},{"cross":{"hidden":"true"}})

    for att_weight, first, second, _, _, _ in attention_pairs:
        stmol.add_cylinder(xyzview, start=first, end=second, cylradius=att_weight, cylColor='red', dashed=False)

    xyzview.addStyle({"elem": "C", "hetflag": True},
                {"stick": {"color": "white", "radius": 0.2}})
    xyzview.addStyle({"hetflag": True},
                        {"stick": {"radius": 0.2}})

    if label_resi:
        for hl_resi in hl_resi_list:
            xyzview.addResLabels({"chain": hl_chain,"resi": hl_resi},
            {"backgroundColor": "lightgray","fontColor": "black","backgroundOpacity": 0.5})

    if label_highest:
        for _, _, chain, res in top_residues:
            xyzview.addResLabels({"chain": chain, "resi": res},
            {"backgroundColor": "lightgray", "fontColor": "black", "backgroundOpacity": 0.5})
            if sidechain_highest:
                xyzview.addStyle({"chain": chain, "resi": res},{"stick": {"radius": 0.2}})
    return xyzview

xyzview = get_3dview(pdb_id)
showmol(xyzview, height=500, width=800)

st.markdown(f"""
Visualize attention weights from protein language models on protein structures.
Currently attention weights for PDB: [{pdb_id}](https://www.rcsb.org/structure/{pdb_id}) from layer: {layer_one}, head: {head_one} above {min_attn} from {selected_model.name.value}
are visualized as red bars. The {n_highest_resis} residues with the highest sum of attention are labeled.
Visualize attention weights on protein structures for the protein language models TAPE-BERT and ZymCTRL.
Pick a PDB ID, layer and head to visualize attention.
""", unsafe_allow_html=True)

chain_dict = {f"{chain.id}": chain for chain in list(structure.get_chains())}
data = []
for att_weight, _ , chain, resi in top_residues:
    res = chain_dict[chain][resi]
    el = (att_weight, f"{res.resname:3}{res.id[1]:0>3}")
    data.append(el)

df = pd.DataFrame(data, columns=['Total attention (disregarding direction)', 'Residue'])
st.markdown(f"The {n_highest_resis} residues with the highest attention sum are labeled in the visualization and listed below:")
st.table(df)

st.markdown("""Clik in to the [Identify Interesting heads](#Identify-Interesting-heads) page to get an overview of attention
            patterns across all layers and heads
            to help you find heads with interesting attention patterns to study here.""")
"""
The attention visualization is inspired by [provis](https://github.com/salesforce/provis#provis-attention-visualizer).
"""