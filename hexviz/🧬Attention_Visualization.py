import pandas as pd
import py3Dmol
import stmol
import streamlit as st
from stmol import showmol

from hexviz.attention import (
    clean_and_validate_sequence,
    get_attention_pairs,
    get_chains,
)
from hexviz.models import Model, ModelType
from hexviz.view import menu_items, select_model, select_pdb, select_protein

st.set_page_config(layout="centered", menu_items=menu_items)
st.title("Attention Visualization on proteins")

for k, v in st.session_state.items():
    st.session_state[k] = v

models = [
    Model(name=ModelType.TAPE_BERT, layers=12, heads=12),
    Model(name=ModelType.ZymCTRL, layers=36, heads=16),
    Model(name=ModelType.PROT_BERT, layers=30, heads=16),
]

with st.expander(
    "Input a PDB id, upload a PDB file or input a sequence", expanded=True
):
    pdb_id = select_pdb()
    uploaded_file = st.file_uploader("2.Upload PDB", type=["pdb"])
    input_sequence = st.text_area(
        "3.Input sequence", "", key="input_sequence", max_chars=400
    )
    sequence, error = clean_and_validate_sequence(input_sequence)
    if error:
        st.error(error)
    pdb_str, structure, source = select_protein(pdb_id, uploaded_file, sequence)
    st.write(f"Visualizing: {source}")

st.sidebar.markdown(
    """
    Configure visualization
    ---
    """
)
chains = get_chains(structure)

if "selected_chains" not in st.session_state:
    st.session_state.selected_chains = chains
selected_chains = st.sidebar.multiselect(
    label="Select Chain(s)", options=chains, key="selected_chains"
)

show_ligands = st.sidebar.checkbox(
    "Show ligands", value=st.session_state.get("show_ligands", True)
)
st.session_state.show_ligands = show_ligands


st.sidebar.markdown(
    """
    Attention parameters
    ---
    """
)
min_attn = st.sidebar.slider(
    "Minimum attention", min_value=0.0, max_value=0.4, value=0.1
)
n_highest_resis = st.sidebar.number_input(
    "Num highest attention resis to label", value=2, min_value=1, max_value=100
)
label_highest = st.sidebar.checkbox("Label highest attention residues", value=True)
sidechain_highest = st.sidebar.checkbox("Show sidechains", value=True)
# TODO add avg or max attention as params


with st.sidebar.expander("Label residues manually"):
    hl_chain = st.selectbox(label="Chain to label", options=selected_chains, index=0)
    hl_resi_list = st.multiselect(
        label="Selected Residues", options=list(range(1, 5000))
    )

    label_resi = st.checkbox(label="Label Residues", value=True)


left, mid, right = st.columns(3)
with left:
    selected_model = select_model(models)
with mid:
    if "selected_layer" not in st.session_state:
        st.session_state["selected_layer"] = 5
    layer_one = st.selectbox(
        "Layer",
        options=[i for i in range(1, selected_model.layers + 1)],
        key="selected_layer",
    )
    layer = layer_one - 1
with right:
    if "selected_head" not in st.session_state:
        st.session_state["selected_head"] = 1
    head_one = st.selectbox(
        "Head",
        options=[i for i in range(1, selected_model.heads + 1)],
        key="selected_head",
    )
    head = head_one - 1

ec_class = ""
if selected_model.name == ModelType.ZymCTRL:
    try:
        ec_class = structure.header["compound"]["1"]["ec"]
    except KeyError:
        pass
    ec_class = st.sidebar.text_input(
        "Enzyme classification number fetched from PDB", ec_class
    )


attention_pairs, top_residues = get_attention_pairs(
    pdb_str=pdb_str,
    chain_ids=selected_chains,
    layer=layer,
    head=head,
    threshold=min_attn,
    model_type=selected_model.name,
    ec_class=ec_class,
    top_n=n_highest_resis,
)

sorted_by_attention = sorted(attention_pairs, key=lambda x: x[0], reverse=True)


def get_3dview(pdb):
    xyzview = py3Dmol.view()
    xyzview.addModel(pdb_str, "pdb")
    xyzview.setStyle({"cartoon": {"color": "spectrum"}})
    stmol.add_hover(xyzview, backgroundColor="black", fontColor="white")

    # Show all ligands as stick (heteroatoms)
    if show_ligands:
        xyzview.addStyle({"hetflag": True}, {"stick": {"radius": 0.2}})

    # If no chains are selected, show all chains
    if selected_chains:
        hidden_chains = [x for x in chains if x not in selected_chains]
        for chain in hidden_chains:
            xyzview.setStyle({"chain": chain}, {"cross": {"hidden": "true"}})
            # Hide ligands for chain too
            xyzview.addStyle(
                {"chain": chain, "hetflag": True}, {"cross": {"hidden": "true"}}
            )

    if len(selected_chains) == 1:
        xyzview.zoomTo({"chain": f"{selected_chains[0]}"})
    else:
        xyzview.zoomTo()

    for att_weight, first, second, _, _, _ in attention_pairs:
        stmol.add_cylinder(
            xyzview,
            start=first,
            end=second,
            cylradius=att_weight,
            cylColor="red",
            dashed=False,
        )

    if label_resi:
        for hl_resi in hl_resi_list:
            xyzview.addResLabels(
                {"chain": hl_chain, "resi": hl_resi},
                {
                    "backgroundColor": "lightgray",
                    "fontColor": "black",
                    "backgroundOpacity": 0.5,
                },
            )

    if label_highest:
        for _, _, chain, res in top_residues:
            xyzview.addResLabels(
                {"chain": chain, "resi": res},
                {
                    "backgroundColor": "lightgray",
                    "fontColor": "black",
                    "backgroundOpacity": 0.5,
                },
            )
            if sidechain_highest:
                xyzview.addStyle(
                    {"chain": chain, "resi": res}, {"stick": {"radius": 0.2}}
                )
    return xyzview


xyzview = get_3dview(pdb_id)
showmol(xyzview, height=500, width=800)

st.markdown(f"""
Visualize attention weights from protein language models on protein structures.
Currently attention weights for PDB: [{pdb_id}](https://www.rcsb.org/structure/{pdb_id}) from layer: {layer_one}, head: {head_one} above {min_attn} from {selected_model.name.value}
are visualized as red bars. The {n_highest_resis} residues with the highest sum of attention are labeled.
Visualize attention weights on protein structures for the protein language models TAPE-BERT, ZymCTRL and ProtBERT.
Pick a PDB ID, layer and head to visualize attention.
""", unsafe_allow_html=True)

chain_dict = {f"{chain.id}": chain for chain in list(structure.get_chains())}
data = []
for att_weight, _ , chain, resi in top_residues:
    res = chain_dict[chain][resi]
    el = (att_weight, f"{res.resname:3}{res.id[1]}")
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