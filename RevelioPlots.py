import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import Polypeptide, is_aa, protein_letters_3to1
import io

# --- Helper Functions ---

def get_color_for_plddt(plddt):
    """Returns a hex color code based on pLDDT score using AlphaFold's scheme."""
    if plddt > 90:
        return "#0053D6"  # Very high (blue)
    elif plddt > 70:
        return "#65CBF3"  # Confident (cyan)
    elif plddt > 50:
        return "#FFDB13"  # Low (yellow)
    else:
        return "#FF7D45"  # Very low (orange)

def get_legend_html():
    """Generates HTML for a color legend."""
    legend_html = """
    <b>pLDDT Confidence Legend:</b>
    <div style="display: flex; flex-wrap: wrap; align-items: center; margin-bottom: 10px; font-size: 14px;">
        <div style="display: flex; align-items: center; margin-right: 15px; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #0053D6; margin-right: 5px; border: 1px solid #ddd;"></div><span>Very high (&gt; 90)</span>
        </div>
        <div style="display: flex; align-items: center; margin-right: 15px; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #65CBF3; margin-right: 5px; border: 1px solid #ddd;"></div><span>Confident (70-90)</span>
        </div>
        <div style="display: flex; align-items: center; margin-right: 15px; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #FFDB13; margin-right: 5px; border: 1px solid #ddd;"></div><span>Low (50-70)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #FF7D45; margin-right: 5px; border: 1px solid #ddd;"></div><span>Very low (&lt; 50)</span>
        </div>
    </div>
    """
    return legend_html

def generate_sequence_figure_html(df):
    """Generates an HTML string for the colored amino acid sequence with a position scale."""
    html = "<div style='font-family: monospace; word-wrap: break-word;'>"
    residues = list(df.to_dict('records'))
    for i in range(0, len(residues), 10):
        group = residues[i:i+10]
        start_res_num = group[0]['Residue']
        html += f"<div style='display: inline-block; vertical-align: top; margin-right: 10px; margin-bottom: 10px;'>"
        html += f"<div style='font-size: 10px; color: grey; padding-left: 2px;'>{start_res_num}</div>"
        html += "<div>"
        for res_data in group:
            aa = res_data['AA']
            plddt = res_data['pLDDT']
            color = get_color_for_plddt(plddt)
            text_color = "white" if plddt > 90 or plddt < 50 else "black"
            tooltip = f"Residue: {res_data['Residue']} | pLDDT: {plddt:.2f}"
            html += f"<span style='background-color: {color}; color: {text_color}; font-size: 16px; padding: 3px 1px; margin: 1px; border-radius: 3px;' title='{tooltip}'>{aa}</span>"
        html += "</div></div>"
    html += "</div>"
    return html

def add_dihedral_angles_to_df(structure, df):
    """Calculates Phi/Psi angles for the structure and merges them into the DataFrame."""
    phi_psi_list = []
    for model in structure:
        for chain in model:
            residues = [res for res in chain if is_aa(res, standard=True)]
            if len(residues) < 2:
                continue
            poly_chain = Polypeptide(residues)
            phi_psi_tuples = poly_chain.get_phi_psi_list()
            for i, res in enumerate(poly_chain):
                res_id = res.get_id()[1]
                phi, psi = phi_psi_tuples[i]
                phi_psi_list.append({
                    'Residue': res_id,
                    'Phi': np.degrees(phi) if phi is not None else None,
                    'Psi': np.degrees(psi) if psi is not None else None,
                })
    if not phi_psi_list:
        df[['Phi', 'Psi']] = None
        return df
    phi_psi_df = pd.DataFrame(phi_psi_list)
    return pd.merge(df, phi_psi_df, on='Residue', how='left')

def generate_ramachandran_plot(df, file_name):
    """Generates an interactive Ramachandran plot."""
    plot_df = df.dropna(subset=['Phi', 'Psi']).copy()
    if plot_df.empty:
        return None
    plot_df['Color'] = plot_df['pLDDT'].apply(get_color_for_plddt)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df['Phi'], y=plot_df['Psi'], mode='markers',
        marker=dict(color=plot_df['Color'], size=8, showscale=False),
        text=plot_df.apply(lambda r: f"Residue: {r['AA']}{r['Residue']}<br>pLDDT: {r['pLDDT']:.2f}<br>Phi: {r['Phi']:.2f}<br>Psi: {r['Psi']:.2f}", axis=1),
        hoverinfo='text', name='Residues'
    ))
    shapes = [
        dict(type="rect", xref="x", yref="y", x0=-180, y0=100, x1=-40, y1=180, fillcolor="rgba(173, 216, 230, 0.2)", layer="below", line_width=0), # Beta
        dict(type="rect", xref="x", yref="y", x0=-160, y0=-70, x1=-30, y1=50, fillcolor="rgba(144, 238, 144, 0.2)", layer="below", line_width=0), # Alpha (R)
        dict(type="rect", xref="x", yref="y", x0=30, y0=0, x1=100, y1=100, fillcolor="rgba(255, 182, 193, 0.2)", layer="below", line_width=0)  # Alpha (L)
    ]
    fig.update_layout(
        title=f"Ramachandran Plot for {file_name}",
        xaxis_title="Phi (Φ) degrees", yaxis_title="Psi (Ψ) degrees",
        xaxis=dict(range=[-180, 180], tickvals=list(range(-180, 181, 60))),
        yaxis=dict(range=[-180, 180], tickvals=list(range(-180, 181, 60))),
        width=600, height=600, showlegend=False, shapes=shapes
    )
    return fig

def calculate_protein_data(file_or_buffer, file_name_for_error_logging=""):
    """
    Parses a CIF file once to extract pLDDT, residue info, and dihedral angles.
    """
    try:
        parser = MMCIFParser(QUIET=True)
        if hasattr(file_or_buffer, 'seek'):
            file_or_buffer.seek(0)
        structure = parser.get_structure("protein", file_or_buffer)
        data = []
        if '_ma_qa_metric_local' in structure.header and structure.header['_ma_qa_metric_local']:
            for row in structure.header['_ma_qa_metric_local']:
                data.append({
                    'Residue': int(row['label_seq_id']), 'pLDDT': float(row['metric_value']),
                    'AA': protein_letters_3to1.get(row.get('label_comp_id', '').upper(), 'X')
                })
        else:
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':
                            for atom in residue:
                                if atom.get_name() == 'CA':
                                    data.append({
                                        'Residue': residue.id[1], 'pLDDT': atom.get_bfactor(),
                                        'AA': protein_letters_3to1.get(residue.get_resname().upper(), 'X')
                                    })
                                    break
        if not data:
            st.warning(f"Could not extract pLDDT values for **{file_name_for_error_logging}**.")
            return None
        df = pd.DataFrame(data).sort_values(by='Residue').reset_index(drop=True)
        return add_dihedral_angles_to_df(structure, df)
    except Exception as e:
        st.error(f"An error occurred while parsing **{file_name_for_error_logging}**: {e}")
        return None

# --- UI Tabs ---

def single_structure_tab():
    st.header("Analyze a Single Protein Structure")
    source_option = st.radio("Choose structure source:", ("Upload a file", "Use an example"), horizontal=True)
    cif_file_source, file_name = None, None
    if source_option == "Upload a file":
        uploaded_file = st.file_uploader("Upload a CIF file (.cif or .mmcif)", type=['cif', 'mmcif'])
        if uploaded_file:
            cif_file_source = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            file_name = uploaded_file.name
    else:
        try:
            example_files = [f for f in os.listdir("examples") if f.endswith(('.cif', '.mmcif'))]
            if not example_files:
                st.warning("No example files found in the 'examples' folder."); return
            selected_file = st.selectbox("Choose an example structure:", [""] + example_files)
            if selected_file:
                cif_file_source = os.path.join("examples", selected_file)
                file_name = selected_file
        except FileNotFoundError:
            st.error("The 'examples' directory was not found."); return
    if cif_file_source and file_name:
        st.info(f"Processing: **{file_name}**")
        protein_df = calculate_protein_data(cif_file_source, file_name)
        if protein_df is not None and not protein_df.empty:
            st.subheader("pLDDT Statistics")
            plddt_values = protein_df['pLDDT']
            col1, col2, col3 = st.columns(3); col1.metric("Mean pLDDT", f"{plddt_values.mean():.2f}"); col2.metric("Median pLDDT", f"{plddt_values.median():.2f}"); col3.metric("Std. Deviation", f"{plddt_values.std():.2f}")
            st.subheader("pLDDT Distribution Plot"); st.plotly_chart(px.box(protein_df, y="pLDDT", points="all", title=f"Distribution of pLDDT Values for {file_name}", labels={"pLDDT": "pLDDT Score"}, template="plotly_white", hover_data=['Residue', 'pLDDT']).update_layout(yaxis_range=[0,100], title_x=0.5), use_container_width=True)
            st.subheader("Confidence-Colored Sequence"); st.markdown(get_legend_html(), unsafe_allow_html=True); st.markdown(generate_sequence_figure_html(protein_df), unsafe_allow_html=True)
            st.subheader("Ramachandran Plot"); st.markdown(get_legend_html(), unsafe_allow_html=True)
            rama_fig = generate_ramachandran_plot(protein_df, file_name)
            if rama_fig: st.plotly_chart(rama_fig, use_container_width=True)
            else: st.warning("Could not generate Ramachandran plot (not enough consecutive standard amino acids).")

def multi_structure_tab():
    st.header("Compare Multiple Protein Structures")
    source_option = st.radio("Choose structure source:", ("Upload files", "Use examples"), horizontal=True, key="multi_source")
    cif_files_info = []
    if source_option == "Upload files":
        uploaded_files = st.file_uploader("Upload CIF files (.cif or .mmcif)", type=['cif', 'mmcif'], accept_multiple_files=True)
        if uploaded_files: cif_files_info = [(io.StringIO(f.getvalue().decode("utf-8")), f.name) for f in uploaded_files]
    else:
        try:
            example_files = [f for f in os.listdir("examples") if f.endswith(('.cif', '.mmcif'))];
            if not example_files: st.warning("No example files found in the 'examples' folder."); return
            selected_files = st.multiselect("Choose example structures to compare:", example_files)
            if selected_files: cif_files_info = [(os.path.join("examples", f), f) for f in selected_files]
        except FileNotFoundError: st.error("The 'examples' directory was not found."); return
    if cif_files_info:
        all_dfs = []
        with st.spinner("Processing files..."):
            for file_source, file_name in cif_files_info:
                protein_df = calculate_protein_data(file_source, file_name)
                if protein_df is not None:
                    protein_df['File'] = file_name; all_dfs.append(protein_df)
        if not all_dfs: st.error("Could not process any of the selected files."); return
        st.subheader("pLDDT Statistics Summary")
        stats_data = {df['File'].iloc[0]: {'Mean': df['pLDDT'].mean(),'Median': df['pLDDT'].median(),'Std Dev': df['pLDDT'].std()} for df in all_dfs}
        st.dataframe(pd.DataFrame.from_dict(stats_data, orient='index').style.format("{:.2f}"), use_container_width=True)
        st.subheader("Comparative pLDDT Distribution Plot"); combined_df = pd.concat(all_dfs, ignore_index=True); st.plotly_chart(px.box(combined_df, x="File", y="pLDDT", color="File", points="all", title="Distribution of pLDDT Values Across Multiple Structures", labels={"pLDDT": "pLDDT Score", "File": "Structure File"}, template="plotly_white", hover_data=['Residue', 'pLDDT']).update_layout(yaxis_range=[0,100], xaxis_title=None, title_x=0.5, showlegend=False), use_container_width=True)
        st.subheader("Per-Structure Analysis")
        # Display the legend ONCE, outside the loop
        st.markdown(get_legend_html(), unsafe_allow_html=True)
        for df in all_dfs:
            file_name = df['File'].iloc[0]
            with st.expander(f"Analysis for: {file_name}"):
                st.markdown("##### Confidence-Colored Sequence")
                st.markdown(generate_sequence_figure_html(df), unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("##### Ramachandran Plot")
                rama_fig = generate_ramachandran_plot(df, file_name)
                if rama_fig:
                    # Add unique key to the chart
                    st.plotly_chart(rama_fig, use_container_width=True, key=f"rama_plot_{file_name}")
                else:
                    st.warning("Could not generate Ramachandran plot for this structure.", key=f"rama_warning_{file_name}")

# --- Main App ---
st.set_page_config(page_title="RevelioPlots", page_icon="🪄", layout="wide")
st.sidebar.title("RevelioPlots")
if os.path.exists('RevelioPlots-logo.png'): st.sidebar.image('RevelioPlots-logo.png', use_container_width=True)
else: st.sidebar.markdown("### pLDDT Visualization Tool")
st.sidebar.markdown("---"); st.sidebar.info("This application analyzes pLDDT scores from protein structure files (.cif) to visualize model confidence.")
if not os.path.exists("examples"): os.makedirs("examples"); st.info("An 'examples' folder has been created. Add .cif files to it to use the example feature.")
tab1, tab2 = st.tabs(["Single Structure Analysis", "Multi-Structure Comparison"])
with tab1: single_structure_tab()
with tab2: multi_structure_tab()