---
title: Hexviz
emoji: 👁️🧬
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: 1.17.0
app_file: ./hexviz/app.py
pinned: false
---
# hexviz
Visualize attention pattern on 3D protein structures

## Install and run

```shell
poetry install

poetry run streamlit run hexviz/streamlit/Attention_On_Structure.py
```

## Export dependcies from poetry
Spaces [require](https://huggingface.co/docs/hub/spaces-dependencies#adding-your-own-dependencies) dependencies in a `requirements.txt` file. Export depencies from poetry's `pyproject.toml` file with:
```shell
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Acknowledgements
This project builds on the attention visualization introduced and developed in 
https://github.com/salesforce/provis#provis-attention-visualizer
