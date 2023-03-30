import py3Dmol
import stmol
import streamlit as st
from stmol import showmol

from hexviz.attention import get_attention_pairs, get_chains, get_structure
from hexviz.models import Model, ModelType

st.title("Attention Visualization on proteins")

"""
Visualize attention weights on protein structures for the protein language models TAPE-BERT and ZymCTRL.
Pick a PDB ID, layer and head to visualize attention.
"""

models = [
    Model(name=ModelType.TAPE_BERT, layers=12, heads=12),
    Model(name=ModelType.ZymCTRL, layers=36, heads=16),
]

selected_model_name = st.selectbox("Select a model", [model.name.value for model in models], index=0)
selected_model = next((model for model in models if model.name.value == selected_model_name), None)

pdb_id = st.sidebar.text_input(
        label="PDB ID",
        value="4RW0",
    )
structure = get_structure(pdb_id)
chains = get_chains(structure)
selected_chains = st.sidebar.multiselect(label="Chain(s)", options=chains, default=chains)

hl_chain = st.sidebar.selectbox(label="Highlight Chain", options=selected_chains, index=0)
hl_resi_list = st.sidebar.multiselect(label="Highlight Residues",options=list(range(1,5000)))

label_resi = st.sidebar.checkbox(label="Label Residues", value=True)


left, right = st.columns(2)
with left:
    layer_one = st.number_input("Layer", value=10, min_value=1, max_value=selected_model.layers)
    layer = layer_one - 1
with right:
    head_one = st.number_input("Head", value=1, min_value=1, max_value=selected_model.heads)
    head = head_one - 1


with st.expander("Attention parameters", expanded=False):
    min_attn = st.slider("Minimum attention", min_value=0.0, max_value=0.4, value=0.1)
    n_pairs = st.number_input("Num attention pairs labeled", value=2, min_value=1, max_value=100)
    label_highest = st.checkbox("Label highest attention pairs", value=True)

    # TODO add avg or max attention as params

    try:
        ec_class = structure.header["compound"]["1"]["ec"]
    except KeyError:
        ec_class = None
    if ec_class and selected_model.name == ModelType.ZymCTRL:
        ec_class = st.text_input("Enzyme classification number fetched from PDB", ec_class)

attention_pairs = get_attention_pairs(pdb_id, chain_ids=selected_chains, layer=layer, head=head, threshold=min_attn, model_type=selected_model.name)

sorted_by_attention = sorted(attention_pairs, key=lambda x: x[0], reverse=True) 
top_n = sorted_by_attention[:n_pairs]

def get_3dview(pdb):
    xyzview = py3Dmol.view(query=f"pdb:{pdb}")
    xyzview.setStyle({"cartoon": {"color": "spectrum"}})
    stmol.add_hover(xyzview, backgroundColor="black", fontColor="white")


    hidden_chains = [x for x in chains if x not in selected_chains]
    for chain in hidden_chains:
        xyzview.setStyle({"chain": chain},{"cross":{"hidden":"true"}})

    for att_weight, first, second, _, _, _ in attention_pairs:
        stmol.add_cylinder(xyzview, start=first, end=second, cylradius=att_weight, cylColor='red', dashed=False)
    
    # get_max_attention(n_pairs)

    if label_resi:
        for hl_resi in hl_resi_list:
            xyzview.addResLabels({"chain": hl_chain,"resi": hl_resi},
            {"backgroundColor": "lightgray","fontColor": "black","backgroundOpacity": 0.5})

    if label_highest:
        for _, _, _, chain, a, b in top_n:
            xyzview.addResLabels({"chain": chain,"resi": a},
            {"backgroundColor": "lightgray","fontColor": "black","backgroundOpacity": 0.5})
            xyzview.addResLabels({"chain": chain,"resi": b},
            {"backgroundColor": "lightgray","fontColor": "black","backgroundOpacity": 0.5})
    return xyzview


xyzview = get_3dview(pdb_id)
showmol(xyzview, height=500, width=800)
st.markdown(f'PDB: [{pdb_id}](https://www.rcsb.org/structure/{pdb_id})', unsafe_allow_html=True)


"""
More models will be added soon. The attention visualization is inspired by [provis](https://github.com/salesforce/provis#provis-attention-visualizer).
"""