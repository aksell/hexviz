import streamlit as st

from hexviz.config import URL

st.markdown(
    f"""
## Protein language models
There has been an explosion of capabilities in natural language processing
models in the last few years.  These architectural advances from NLP have proven
to work very well for protein sequences, and we now have protein language models
(pLMs) that can generate novel functional proteins sequences
[ProtGPT2](https://www.nature.com/articles/s42256-022-00499-z) and auto-encoding
models that excel at capturing biophysical features of protein sequences
[ProtTrans](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3).

For an introduction to protein language models for protein design check out
[Controllable protein design with language
models](https://www.nature.com/articles/s42256-022-00499-z).

## Interpreting protein language models by visualizing attention patterns
With these impressive capabilities it is natural to ask what protein language
models are learning and how they work -- we want to **interpret** the models.
In natural language processing **attention analysis** has proven to be a useful
tool for interpreting transformer model internals see fex ([Abnar et al.
2020](https://arxiv.org/abs/2005.00928v2)).  [BERTology meets
biology](https://arxiv.org/abs/2006.15222) provides a thorough introduction to
how we can analyze Transformer protein models through the lens of attention,
they show exciting findings such as: 
> Attention: (1) captures the folding
> structure of proteins, connecting amino acids that are far apart in the
> underlying sequence, but spatially close in the three-dimensional structure, (2)
> targets binding sites, a key functional component of proteins, and (3) focuses
> on progressively more complex biophysical properties with increasing layer depth

Most existing tools for analyzing and visualizing attention patterns focus on
models trained on text ([BertViz](https://github.com/jessevig/bertviz),
[exBERT], [exBERT](https://exbert.net/)). It can be hard to analyze protein
sequences using these tools as we don't have any intuitive understand about the
protein language when looking at an amino acid sequence in the same way we do
for natural language.  Experts studying proteins do have an understanding of
proteins, but it is mostly in in the context of a protein's structure, not its
sequence. Can we build a tool for analyzing attention patterns that can leverage
expert's knowledge of protein structure to understand what pLMs learn?

BERTology meets biology shows how visualizing attention patterns in the context
of protein structure can facilitate novel discoveries about what models learn.
[**Hexviz**](https://huggingface.co/spaces/aksell/hexviz) builds on this, and is
a tool to simplify analyzing attention patterns in the context of protein
structure. We hope this can enable domain experts to explore and interpret the
knowledge contained in pLMs.

## How to use Hexviz
There are two views:
1. <a href="{URL}Attention_Visualization" target="_self">🧬Attention Visualization</a> Shows attention weights from a single head as red bars between residues on a protein structure.
2. <a href="{URL}Identify_Interesting_Heads" target="_self">🗺️Identify Interesting Heads</a> Plots attention weights between residues as a heatmap for each head in the model.

The first view is the meat of the application and is where you can investigate
how attention patterns map onto the structure of a protein you're interested in.
Use the second view to narrow down to a few heads that you want to investigate
attention patterns from in detail.  pLM are large and can have many heads, as an
example ProtBERT with it's 30 layers and 16 heads has 480 heads, so we need a
way to identify heads with patterns we're interested in.

The second view is a customizable heatmap plot of attention between residue for
all heads and layers in a model. From here it is possible to identify heads that
specialize in a particular attention pattern, such as:
1. Vertical lines: Paying attention so a single or a few residues
2. Diagonal: Attention to the same residue or residues in front or behind the current residue.
3. Block attention: Attention is segmented so parts of the sequence are attended to by one part of the sequence.
4. Heterogeneous: More complex attention patterns that are not easily categorized.
TODO: Add examples of attention patterns

Read more about attention patterns in fex [Revealing the dark secrets of
BERT](https://arxiv.org/abs/1908.08593).

## Protein Language models in Hexviz
Hexviz currently supports the following models:
1. [ProtBERT](https://huggingface.co/Rostlab/prot_bert_bfd)
2. [ZymCTRL](https://huggingface.co/nferruz/ZymCTRL)
3. [TapeBert](https://github.com/songlab-cal/tape/blob/master/tape/models/modeling_bert.py) - a nickname coined in BERTology meets biology for the Bert Base model pre-trained on Pfam in [TAPE](https://www.biorxiv.org/content/10.1101/676825v1). TapeBert is used extensively in BERTOlogy meets biology.
4. [ProtT5 half](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)

## FAQ
1. I can't see any attention- "bars" in the visualization, what is wrong? -> Lower the `minimum attention`.
2. How are sequences I input folded? -> Using https://esmatlas.com/resources?action=fold
""",
    unsafe_allow_html=True,
)
