import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
from Bio.PDB import MMCIFParser
import io

# --- Helper Function ---

def calculate_plddt_from_cif(file_or_buffer, file_name_for_error_logging=""):
    """
    Parses a CIF file (from a path or an in-memory buffer) to extract pLDDT values.
    It tries the '_ma_qa_metric_local' field first, falling back to B-factors.

    Args:
        file_or_buffer: A file path (str) or a file-like object (e.g., BytesIO).
        file_name_for_error_logging (str): The name of the file for clearer error messages.

    Returns:
        list: A list of pLDDT values, or None if an error occurs.
    """
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", file_or_buffer)
        
        # Try to get pLDDT from the standard AlphaFold field first
        try:
            plddt_values = [float(row[4]) for row in structure.header['_ma_qa_metric_local']]
            if plddt_values:
                return plddt_values
        except (KeyError, IndexError):
            pass # Common case, so we'll try B-factors next

        # Fallback to B-factors if the standard field is not found
        plddt_values = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard amino acid residues
                        for atom in residue:
                            if atom.get_name() == 'CA':
                                plddt_values.append(atom.get_bfactor())
                                break
        
        if plddt_values:
            return plddt_values
        else:
            st.warning(f"Could not extract pLDDT values for **{file_name_for_error_logging}**.")
            return None

    except Exception as e:
        st.error(f"An error occurred while parsing **{file_name_for_error_logging}**: {e}")
        return None

# --- UI and Plotting Functions for Streamlit ---

def single_structure_tab():
    """Handles the UI and logic for the 'Single Structure' tab."""
    st.header("Analyze a Single Protein Structure")

    # Determine the source of the file
    source_option = st.radio("Choose structure source:", ("Upload a file", "Use an example"), horizontal=True)

    cif_file_source = None
    file_name = None

    if source_option == "Upload a file":
        uploaded_file = st.file_uploader("Upload a CIF file (.cif or .mmcif)", type=['cif', 'mmcif'])
        if uploaded_file:
            cif_file_source = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            file_name = uploaded_file.name
    else: # Use an example
        try:
            example_files = [f for f in os.listdir("examples") if f.endswith(('.cif', '.mmcif'))]
            if not example_files:
                st.warning("No example files found in the 'examples' folder.")
                return
            selected_file = st.selectbox("Choose an example structure:", [""] + example_files)
            if selected_file:
                cif_file_source = os.path.join("examples", selected_file)
                file_name = selected_file
        except FileNotFoundError:
            st.error("The 'examples' directory was not found. Please create it to use this feature.")
            return


    if cif_file_source and file_name:
        st.info(f"Processing: **{file_name}**")
        plddt_values = calculate_plddt_from_cif(cif_file_source, file_name)

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

            # --- Plotting with Plotly ---
            st.subheader("pLDDT Distribution Plot")
            df = pd.DataFrame({'pLDDT': plddt_values})
            
            fig = px.box(
                df, 
                y="pLDDT",
                points="all", # Show all data points
                title=f"Distribution of pLDDT Values for {file_name}",
                labels={"pLDDT": "pLDDT Score"},
                template="plotly_white",
                hover_data=df.columns
            )
            
            fig.update_layout(
                yaxis_range=[0,100],
                title_x=0.5,
                font=dict(size=14)
            )
            st.plotly_chart(fig, use_container_width=True)


def multi_structure_tab():
    """Handles the UI and logic for the 'Multi-Structure' tab."""
    st.header("Compare Multiple Protein Structures")

    source_option = st.radio("Choose structure source:", ("Upload files", "Use examples"), horizontal=True, key="multi_source")
    
    cif_files = []
    if source_option == "Upload files":
        uploaded_files = st.file_uploader("Upload CIF files (.cif or .mmcif)", type=['cif', 'mmcif'], accept_multiple_files=True)
        if uploaded_files:
            cif_files = [(io.StringIO(f.getvalue().decode("utf-8")), f.name) for f in uploaded_files]
    else: # Use examples
        try:
            example_files = [f for f in os.listdir("examples") if f.endswith(('.cif', '.mmcif'))]
            if not example_files:
                st.warning("No example files found in the 'examples' folder.")
                return
            selected_files = st.multiselect("Choose example structures to compare:", example_files)
            if selected_files:
                cif_files = [(os.path.join("examples", f), f) for f in selected_files]
        except FileNotFoundError:
            st.error("The 'examples' directory was not found. Please create it to use this feature.")
            return

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
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

        # --- Plotting with Plotly ---
        st.subheader("Comparative pLDDT Distribution Plot")
        
        fig = px.box(
            df,
            x="File",
            y="pLDDT",
            color="File", # Color each box differently
            points="all",
            title="Distribution of pLDDT Values Across Multiple Structures",
            labels={"pLDDT": "pLDDT Score", "File": "Structure File"},
            template="plotly_white"
        )

        fig.update_layout(
            yaxis_range=[0,100],
            xaxis_title=None,
            title_x=0.5,
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Main App ---
st.set_page_config(page_title="RevelioPlots", page_icon="🪄", layout="wide")

st.sidebar.title("RevelioPlots")

# Use a local file path for the logo
logo_path = 'RevelioPlots-logo.png'
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True) # <-- CORRECTED PARAMETER
else:
    st.sidebar.markdown("### pLDDT Visualization Tool")

st.sidebar.markdown("---")
st.sidebar.info("This application analyzes the pLDDT scores from protein structure files (.cif) to visualize model confidence.")

# Create an 'examples' directory if it doesn't exist
if not os.path.exists("examples"):
    os.makedirs("examples")
    st.info("An 'examples' folder has been created. Add some .cif files to it to use the example feature.")

# --- Main App Tabs ---
tab1, tab2 = st.tabs(["Single Structure Analysis", "Multi-Structure Comparison"])

with tab1:
    single_structure_tab()

with tab2:
    multi_structure_tab()