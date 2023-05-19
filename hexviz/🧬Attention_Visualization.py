import re

import numpy as np
import pandas as pd
import py3Dmol
import stmol
import streamlit as st
from stmol import showmol

from hexviz.attention import clean_and_validate_sequence, get_attention_pairs, get_chains
from hexviz.config import URL
from hexviz.ec_number import ECNumber
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

if "show_ligands" not in st.session_state:
    st.session_state.show_ligands = True
show_ligands = st.sidebar.checkbox("Show ligands", key="show_ligands")
if "color_protein" not in st.session_state:
    st.session_state.color_protein = True
color_protein = st.sidebar.checkbox("Color protein", key="color_protein")


st.sidebar.markdown(
    """
    Attention parameters
    ---
    """
)
min_attn = st.sidebar.slider("Minimum attention", min_value=0.0, max_value=0.4, value=0.1)
n_highest_resis = st.sidebar.number_input(
    "Num highest attention resis to label", value=2, min_value=1, max_value=100
)
label_highest = st.sidebar.checkbox("Label highest attention residues", value=True)
sidechain_highest = st.sidebar.checkbox("Show sidechains", value=True)


with st.sidebar.expander("Label residues manually"):
    hl_chain = st.selectbox(label="Chain to label", options=selected_chains, index=0)
    hl_resi_list = st.multiselect(label="Selected Residues", options=list(range(1, 5000)))

    label_resi = st.checkbox(label="Label Residues", value=True)


left, mid, right = st.columns(3)
with left:
    selected_model = select_model(models)
with mid:
    if "selected_layer" not in st.session_state:
        st.session_state["selected_layer"] = 5
    layer_one = (
        st.selectbox(
            "Layer",
            options=[i for i in range(1, selected_model.layers + 1)],
            key="selected_layer",
        )
        or 5
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
            "Please enter a valid Enzyme Commission number in the format of 4 integers separated by periods (e.g., 1.2.3.21)"
        )

    if ec_number:
        if selected_chains:
            shown_chains = [ch for ch in structure.get_chains() if ch.id in selected_chains]
        else:
            shown_chains = list(structure.get_chains())

        EC_tags = []
        colors = ["blue", "green", "orange", "red"]
        radius = 1
        EC_numbers = ec_number.split(".")
        for ch in shown_chains:
            first_residues = []
            i = 1
            while len(first_residues) < 2:
                try:
                    first_residues.append(ch[i]["CA"].coord.tolist())
                except KeyError:
                    pass
                i += 1
            res_1, res_2 = first_residues

            # Calculate the vector from res_1 to res_2
            vector = [res_2[i] - res_1[i] for i in range(3)]

            # Reverse the vector
            reverse_vector = [-v for v in vector]

            # Normalize the reverse vector
            reverse_vector_normalized = np.array(reverse_vector) / np.linalg.norm(reverse_vector)
            coordinates = [
                [res_1[j] + i * 2 * radius * reverse_vector_normalized[j] for j in range(3)]
                for i in range(4)
            ]
            EC_tag = [
                ECNumber(number=num, coordinate=coord, color=color, radius=radius)
                for num, coord, color in zip(EC_numbers, coordinates, colors)
            ]
            EC_tags.append(EC_tag)

        EC_colored = [f":{color}[{num}]" for num, color in zip(EC_numbers, colors)]
        st.sidebar.write("Visualized as colored spheres: " + ".".join(EC_colored))


attention_pairs, top_residues = get_attention_pairs(
    pdb_str=pdb_str,
    chain_ids=selected_chains,
    layer=layer,
    head=head,
    threshold=min_attn,
    model_type=selected_model.name,
    top_n=n_highest_resis,
    ec_numbers=EC_tags if ec_number else None,
)

sorted_by_attention = sorted(attention_pairs, key=lambda x: x[0], reverse=True)


def get_3dview(pdb):
    xyzview = py3Dmol.view()
    xyzview.addModel(pdb_str, "pdb")
    xyzview.setStyle({"cartoon": {"color": "spectrum" if color_protein else "white"}})
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
            xyzview.addStyle({"chain": chain, "hetflag": True}, {"cross": {"hidden": "true"}})

    if len(selected_chains) == 1:
        xyzview.zoomTo({"chain": f"{selected_chains[0]}"})
    else:
        xyzview.zoomTo()

    for att_weight, first, second in attention_pairs:
        stmol.add_cylinder(
            xyzview,
            start=first,
            end=second,
            cylradius=att_weight,
            cylColor="red",
            dashed=False,
        )

    if selected_model.name == ModelType.ZymCTRL and ec_number:
        for EC_tag in EC_tags:
            for EC_num in EC_tag:
                stmol.add_sphere(
                    xyzview,
                    spcenter=EC_num.coordinate,
                    radius=EC_num.radius,
                    spColor=EC_num.color,
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
        for _, chain, res in top_residues:
            one_indexed_res = res + 1
            xyzview.addResLabels(
                {"chain": chain, "resi": one_indexed_res},
                {
                    "backgroundColor": "lightgray",
                    "fontColor": "black",
                    "backgroundOpacity": 0.5,
                },
            )
            if sidechain_highest:
                xyzview.addStyle({"chain": chain, "resi": res}, {"stick": {"radius": 0.2}})
    return xyzview


xyzview = get_3dview(pdb_id)
showmol(xyzview, height=500, width=800)

st.markdown(
    f"""
Pick a PDB ID, layer and head to visualize attention from the selected protein language model ({selected_model.name.value}).
""",
    unsafe_allow_html=True,
)

chain_dict = {f"{chain.id}": list(chain.get_residues()) for chain in list(structure.get_chains())}
data = []
for att_weight, chain, resi in top_residues:
    try:
        res = chain_dict[chain][resi]
    except KeyError:
        continue
    el = (att_weight, f"{res.resname:3}{res.id[1]}({chain})")
    data.append(el)

df = pd.DataFrame(data, columns=["Total attention to", "Residue"])
st.markdown(
    f"The {n_highest_resis} residues (per chain) with the highest attention to them are labeled in the visualization and listed here:"
)
st.table(df)

st.markdown(
    f"""
### Check out the other pages
<a href="{URL}Identify_Interesting_Heads" target="_self">🗺️Identify Interesting Heads</a> gives a
 bird's eye view of attention patterns for a model.
This can help you pick what specific attention heads to look at for your protein.

<a href="{URL}Documentation" target="_self">📄Documentation</a> has information on protein 
language models, attention analysis and hexviz.""",
    unsafe_allow_html=True,
)

"""
The attention visualization is inspired by [provis](https://github.com/salesforce/provis#provis-attention-visualizer).
"""
