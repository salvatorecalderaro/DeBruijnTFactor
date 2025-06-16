import streamlit as st
from PIL import Image
from Bio import SeqIO
import io
import numpy as np
import joblib
from prot2graph import create_graphs

# --- Constants ---
LOGO_PATH = "GraphTFactor_logo.png"  # Ensure logo.png is in the same directory
MODEL_PATH = "graph_tfactor_model.pkl"  # Path to your trained model

# --- Page Configuration ---
st.set_page_config(page_title="GraphTFactor", page_icon=LOGO_PATH)

# --- Sidebar with Logo and Citation ---
with st.sidebar:
    try:
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=200)
    except Exception as e:
        st.warning(f"⚠️ Could not load logo: {e}")

    st.markdown("""
    ---
    ## 📄 Reference

    This app uses a **Graph Neural Network (GNN)** to predict transcription factor presence from protein sequences.

    **Citation:**  
    Amato, D., Calderaro, S., Lo Bosco, G., Vella, F., & Rizzo, R. (2025). Proteins Transcription Factor Prediction Using Graph Neural Networks. In International Meeting on Computational Intelligence Methods for Bioinformatics and Biostatistics (pp. 15-27). Springer, Cham.
    [📄 DOI](https://link.springer.com/chapter/10.1007/978-3-031-89704-7_2)

    ---
    """)

# --- Title ---
st.title("🧬 GraphTFactor")
st.markdown("**Predict Transcription Factor Presence from Protein Sequences**")




input_method = st.radio("Input Method", ["Upload FASTA File", "Paste Sequence"])
sequence = ""

if input_method == "Upload FASTA File":
    fasta_file = st.file_uploader("Upload your protein FASTA file", type=["fasta", "fa", "txt"])
    if fasta_file:
        try:
            record = next(SeqIO.parse(io.StringIO(fasta_file.getvalue().decode("utf-8")), "fasta"))
            sequence = str(record.seq)
            st.success(f"✅ Loaded sequence: `{record.id}`")
        except Exception as e:
            st.error(f"❌ Error reading FASTA: {e}")

elif input_method == "Paste Sequence":
    pasted_seq = st.text_area("Paste your protein sequence here", height=150)
    if pasted_seq:
        sequence = pasted_seq.strip()


# --- Predict Button ---
if sequence:
    if st.button("🔍 Predict Transcription Factor Presence"):
        try:
            graphs = create_graphs(sequence)
            """
            if prediction == 1:
                st.success("✅ This sequence **contains** a transcription factor.")
            else:
                st.warning("❌ This sequence **does not contain** a transcription factor.")
            """
        except Exception as e:
            st.error(f"Prediction failed: {e}")