import re

import py3Dmol
import stmol
import streamlit as st

from hexviz.attention import (
    clean_and_validate_sequence,
    get_attention,
    get_attention_pairs,
    res_to_1letter,
)
from hexviz.models import Model, ModelType
from hexviz.view import (
    menu_items,
    select_heads_and_layers,
    select_model,
    select_pdb,
    select_protein,
)

st.set_page_config(layout="wide", menu_items=menu_items)
st.title("Bird's Eye View of attention heads")


for k, v in st.session_state.items():
    st.session_state[k] = v

models = [
    Model(name=ModelType.TAPE_BERT, layers=12, heads=12),
    Model(name=ModelType.ZymCTRL, layers=36, heads=16),
    Model(name=ModelType.PROT_BERT, layers=30, heads=16),
    Model(name=ModelType.PROT_T5, layers=24, heads=32),
]

with st.expander("Input a PDB id, upload a PDB file or input a sequence", expanded=True):
    pdb_id = select_pdb() or "2FZ5"
    uploaded_file = st.file_uploader("2.Upload PDB", type=["pdb"])
    input_sequence = st.text_area("3.Input sequence", "", key="input_sequence", max_chars=400)
    sequence, error = clean_and_validate_sequence(input_sequence)
    if error:
        st.error(error)
    pdb_str, structure, source = select_protein(pdb_id, uploaded_file, sequence)
    st.write(f"Visualizing: {source}")

selected_model = select_model(models)


if "viewer_width" not in st.session_state:
    st.session_state.viewer_width = 1200
viewer_width = st.sidebar.number_input(
    label="Protein viewer width (px)",
    min_value=0,
    key="viewer_width",
)

chains = list(structure.get_chains())
chain_ids = [chain.id for chain in chains]
if "selected_chain" not in st.session_state:
    st.session_state.selected_chain = chain_ids[0] if chain_ids else None
chain_selection = st.sidebar.selectbox(
    label="Select Chain",
    options=chain_ids,
    key="selected_chain",
)

selected_chain = next(chain for chain in chains if chain.id == chain_selection)

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
            """Please enter a valid Enzyme Commission number in the format of 4
            integers separated by periods (e.g., 1.2.3.21)"""
        )


min_attn = st.sidebar.slider("Minimum attention", min_value=0.0, max_value=0.4, value=0.1)
if "show_ligands" not in st.session_state:
    st.session_state.show_ligands = True
show_ligands = st.sidebar.checkbox("Show ligands", key="show_ligands")

with st.sidebar.expander("Highlight residues"):
    st.write("Residue will be highlighted in yellow")
    hl_resi_list = st.multiselect(label="Selected Residues", options=list(range(1, 5000)))
    highlight_resi = st.checkbox(label="Highlight residues", value=True)
    label_resi = st.checkbox(label="Label residue names", value=False)
layer_sequence, head_sequence = select_heads_and_layers(st.sidebar, selected_model)
# TODO add slider for widht of grid


residues = [res for res in selected_chain.get_residues()]
sequence = res_to_1letter(residues)

attention, tokens = get_attention(
    sequence=sequence,
    model_type=selected_model.name,
    ec_number=ec_number,
)

grid_rows = len(layer_sequence)
grid_cols = len(head_sequence)
cell_width = viewer_width / grid_cols
viewer_height = int(cell_width * grid_rows)

xyzview = py3Dmol.view(
    width=viewer_width,
    height=viewer_height,
    viewergrid=(grid_rows, grid_cols),
)

xyzview.addModel(pdb_str, "pdb")
xyzview.zoomTo()

for row, layer in enumerate(layer_sequence):
    for col, head in enumerate(head_sequence):
        attention_pairs, top_residues = get_attention_pairs(
            pdb_str=pdb_str,
            chain_ids=None,
            layer=layer,
            head=head,
            threshold=min_attn,
            model_type=selected_model.name,
            top_n=1,
            ec_numbers=None,
        )

        for att_weight, first, second in attention_pairs:
            cylradius = att_weight
            cylColor = "red"
            dashed = False
            xyzview.addCylinder(
                {
                    "start": {"x": first[0], "y": first[1], "z": first[2]},
                    "end": {"x": second[0], "y": second[1], "z": second[2]},
                    "radius": cylradius,
                    "fromCap": True,
                    "toCap": True,
                    "color": cylColor,
                    "dashed": dashed,
                },
                viewer=(row, col),
            )


xyzview.setStyle({"cartoon": {"color": "white"}})
if highlight_resi:
    for res in hl_resi_list:
        xyzview.setStyle({"resi": res}, {"cartoon": {"color": "yellow"}})
if label_resi:
    for hl_resi in hl_resi_list:
        xyzview.addResLabels(
            {"resi": hl_resi},
            {
                "backgroundColor": "lightgray",
                "fontColor": "black",
                "backgroundOpacity": 0.5,
            },
        )
if show_ligands:
    xyzview.addStyle({"hetflag": True}, {"stick": {"radius": 0.2}})

stmol.showmol(xyzview, height=viewer_height, width=viewer_width)
