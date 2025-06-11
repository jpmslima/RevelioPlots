import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import MMCIFParser
import io

# --- Helper Functions (Combined and Adapted from both scripts) ---

def calculate_plddt_from_cif(file_or_buffer, file_name_for_error_logging=""):
    """
    Parses a CIF file (from a path or an in-memory buffer) to extract pLDDT values.
    It tries to get pLDDTs from the '_ma_qa_metric_local' field first,
    and falls back to B-factors if the first field is not available.

    Args:
        file_or_buffer: A file path (str) or a file-like object (e.g., BytesIO).
        file_name_for_error_logging (str): The name of the file for clearer error messages.

    Returns:
        list: A list of pLDDT values, or None if an error occurs.
    """
    try:
        parser = MMCIFParser(QUIET=True)
        # The get_structure method can handle both file paths and file-like objects
        structure = parser.get_structure("protein", file_or_buffer)
        
        # Try to get pLDDT from _ma_qa_metric_local first (standard for modern AlphaFold CIFs)
        try:
            plddt_values = [float(row[4]) for row in structure.header['_ma_qa_metric_local']]
            if plddt_values:
                return plddt_values
        except (KeyError, IndexError):
            # This is a common case, so we just pass and try the next method
            pass

        # If not found or empty, fall back to B-factors (often used in older or converted files)
        plddt_values = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard residues
                        for atom in residue:
                            if atom.get_name() == 'CA':
                                plddt_values.append(atom.get_bfactor())
                                break  # Move to the next residue once CA is found
        
        if plddt_values:
            return plddt_values
        else:
            st.warning(f"Could not extract pLDDT values for {file_name_for_error_logging} from either '_ma_qa_metric_local' or B-factors.")
            return None

    except Exception as e:
        st.error(f"An error occurred while parsing {file_name_for_error_logging}: {e}")
        return None

# --- UI and Plotting Functions for Streamlit ---

def single_structure_tab():
    """Handles the UI and logic for the 'Single Structure' tab."""
    st.header("Analyze a Single Protein Structure")

    cif_file = None
    if st.session_state.use_examples:
        example_files = [f for f in os.listdir("examples") if f.endswith(('.cif', '.mmcif'))]
        if not example_files:
            st.warning("No example files found in the 'examples' folder.")
            return
        selected_file = st.selectbox("Choose an example structure:", [""] + example_files)
        if selected_file:
            cif_file = os.path.join("examples", selected_file)
            file_name = selected_file
    else:
        uploaded_file = st.file_uploader("Upload a CIF file (.cif or .mmcif)", type=['cif', 'mmcif'])
        if uploaded_file is not None:
            # To make it compatible with the parser, we read it into a buffer
            cif_file = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            file_name = uploaded_file.name

    if cif_file:
        st.info(f"Processing: **{file_name}**")
        plddt_values = calculate_plddt_from_cif(cif_file, file_name)

        if plddt_values:
            # --- Calculate Statistics ---
            mean_plddt = np.mean(plddt_values)
            median_plddt = np.median(plddt_values)
            std_dev_plddt = np.std(plddt_values)
            
            st.subheader("pLDDT Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean pLDDT", f"{mean_plddt:.2f}")
            col2.metric("Median pLDDT", f"{median_plddt:.2f}")
            col3.metric("Std. Deviation", f"{std_dev_plddt:.2f}")

            # --- Plotting ---
            st.subheader("pLDDT Distribution Plot")
            fig, ax = plt.subplots(figsize=(8, 7))
            
            sns.boxplot(y=plddt_values, color="skyblue", width=0.4, ax=ax)
            sns.stripplot(y=plddt_values, color="black", size=2.5, jitter=0.2, alpha=0.5, ax=ax)

            stats_text = (f"Mean: {mean_plddt:.2f}\n"
                          f"Median: {median_plddt:.2f}\n"
                          f"Std Dev: {std_dev_plddt:.2f}")
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

            ax.set_title(f"Distribution of pLDDT Values for {file_name}", fontsize=14)
            ax.set_ylabel("pLDDT Score", fontsize=12)
            ax.set_xlabel("Structure", fontsize=12)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks([]) # Hide x-axis ticks for a single plot
            
            plt.tight_layout()
            st.pyplot(fig)


def multi_structure_tab():
    """Handles the UI and logic for the 'Multi-Structure' tab."""
    st.header("Compare Multiple Protein Structures")

    cif_files = []
    if st.session_state.use_examples:
        example_files = [f for f in os.listdir("examples") if f.endswith(('.cif', '.mmcif'))]
        if not example_files:
            st.warning("No example files found in the 'examples' folder.")
            return
        selected_files = st.multiselect("Choose example structures to compare:", example_files)
        if selected_files:
            cif_files = [(os.path.join("examples", f), f) for f in selected_files]
    else:
        uploaded_files = st.file_uploader("Upload CIF files (.cif or .mmcif)", type=['cif', 'mmcif'], accept_multiple_files=True)
        if uploaded_files:
            cif_files = [(io.StringIO(f.getvalue().decode("utf-8")), f.name) for f in uploaded_files]

    if cif_files:
        all_plddts = []
        stats_data = {}
        
        with st.spinner("Processing files..."):
            for file_path_or_buffer, file_name in cif_files:
                plddts = calculate_plddt_from_cif(file_path_or_buffer, file_name)
                if plddts:
                    all_plddts.extend([{'File': file_name, 'pLDDT': plddt} for plddt in plddts])
                    stats_data[file_name] = {
                        'Mean': np.mean(plddts),
                        'Median': np.median(plddts),
                        'Std Dev': np.std(plddts)
                    }
        
        if not all_plddts:
            st.error("Could not process any of the selected files.")
            return

        df = pd.DataFrame(all_plddts)
        stats_df = pd.DataFrame.from_dict(stats_data, orient='index')

        # --- Display Statistics Table ---
        st.subheader("pLDDT Statistics Summary")
        st.dataframe(stats_df.style.format("{:.2f}"))

        # --- Plotting ---
        st.subheader("Comparative pLDDT Distribution Plot")
        
        num_files = len(df['File'].unique())
        fig_width = max(8, num_files * 1.5)
        fig, ax = plt.subplots(figsize=(fig_width, 7))

        sns.boxplot(x='File', y='pLDDT', data=df, palette='viridis', ax=ax)

        # Add mean value text above each box
        file_order = df['File'].unique()
        for i, file_name in enumerate(file_order):
            mean_val = stats_data[file_name]['Mean']
            ax.text(i, 102, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

        ax.set_title("Distribution of pLDDT Values Across Multiple Structures", fontsize=14)
        ax.set_ylabel("pLDDT Score", fontsize=12)
        ax.set_xlabel("Structure File", fontsize=12)
        ax.set_ylim(0, 108)  # Adjust ylim to make space for text
        plt.xticks(rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)


# --- Main App ---
st.set_page_config(page_title="RevelioPlots", layout="wide")

st.sidebar.title("RevelioPlots")

if os.path.exists('RevelioPlots-logo.png'):
    st.sidebar.image('RevelioPlots-logo.png', use_column_width=True)
else:
    # A fallback if the logo file is not found
    st.sidebar.markdown("### pLDDT Visualization Tool")


st.sidebar.header("Options")
# Initialize session state for the checkbox
if 'use_examples' not in st.session_state:
    st.session_state.use_examples = False

# The checkbox will now update the session state
st.sidebar.checkbox("Use example files", key="use_examples",
                    help="If checked, you can select from pre-loaded files in the 'examples' folder.")


# Create an 'examples' directory if it doesn't exist (for first-time users)
if not os.path.exists("examples"):
    os.makedirs("examples")
    st.info("An 'examples' folder has been created. Please add some .cif files to it to use the example feature.")


tab1, tab2 = st.tabs(["Single Structure Analysis", "Multi-Structure Comparison"])

with tab1:
    single_structure_tab()

with tab2:
    multi_structure_tab()