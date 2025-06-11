import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import MMCIFParser

def calculate_plddt_stats(cif_file):
    """
    Parses a single CIF file to extract pLDDT values.

    Args:
        cif_file (str): The path to the CIF file.

    Returns:
        list: A list of pLDDT values, or None if an error occurs.
    """
    if not os.path.exists(cif_file):
        print(f"Warning: File not found at {cif_file}. Skipping.")
        return None

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
        plddt_values = []

        # Try _ma_qa_metric_local first
        try:
            # We need to access the raw dictionary for this
            plddt_values = [float(row[4]) for row in structure.header['_ma_qa_metric_local']]
            if plddt_values:
                # print(f"Info: Extracted pLDDT from _ma_qa_metric_local for {os.path.basename(cif_file)}.")
                return plddt_values
        except KeyError:
            # print(f"Info: '_ma_qa_metric_local' not found for {os.path.basename(cif_file)}, trying B-factors.")
            pass # Continue to B-factor extraction

        # Fallback to B-factors (using CA atoms)
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ': # Standard residues
                        for atom in residue:
                            if atom.get_name() == 'CA':
                                plddt_values.append(atom.get_bfactor())
                                break
        
        if not plddt_values:
             print(f"Warning: Could not extract pLDDT values from {os.path.basename(cif_file)}.")
             return None

        # print(f"Info: Extracted {len(plddt_values)} pLDDT values from B-factors for {os.path.basename(cif_file)}.")
        return plddt_values

    except Exception as e:
        print(f"Error parsing CIF file {os.path.basename(cif_file)}: {e}")
        return None

def process_multiple_cifs(cif_files):
    """
    Processes multiple CIF files, extracts pLDDT, and calculates stats.

    Args:
        cif_files (list): A list of paths to CIF files.

    Returns:
        tuple: A pandas DataFrame with pLDDT data and a dictionary with stats,
               or (None, None) if no data could be processed.
    """
    all_plddts = []
    stats_data = {}
    
    for cif_file in cif_files:
        file_name = os.path.basename(cif_file)
        plddts = calculate_plddt_stats(cif_file)

        if plddts:
            # Store for DataFrame
            for plddt in plddts:
                all_plddts.append({'File': file_name, 'pLDDT': plddt})
            
            # Calculate and store stats
            mean_plddt = np.mean(plddts)
            median_plddt = np.median(plddts)
            std_dev_plddt = np.std(plddts)
            stats_data[file_name] = {
                'Mean': mean_plddt,
                'Median': median_plddt,
                'Std Dev': std_dev_plddt
            }

    if not all_plddts:
        print("Error: Could not process any of the input CIF files.")
        return None, None

    df = pd.DataFrame(all_plddts)
    return df, stats_data

def plot_multiple_plddt_distributions(df, stats_data):
    """
    Generates a box plot for multiple pLDDT distributions.

    Args:
        df (pd.DataFrame): DataFrame with 'File' and 'pLDDT' columns.
        stats_data (dict): Dictionary with statistics for each file.
    """
    if df is None or df.empty:
        print("No data available for plotting.")
        return

    print("\n--- Statistics Summary ---")
    print(f"{'File':<30} | {'Mean':<7} | {'Median':<7} | {'Std Dev':<7}")
    print("-" * 60)
    for file_name, stats in stats_data.items():
        print(f"{file_name:<30} | {stats['Mean']:<7.2f} | {stats['Median']:<7.2f} | {stats['Std Dev']:<7.2f}")
    print("-" * 60)

    num_files = len(df['File'].unique())
    plt.figure(figsize=(max(8, num_files * 1.5), 7)) # Adjust width based on number of files

    # Box plot
    sns.boxplot(x='File', y='pLDDT', data=df, palette='viridis')
    
    # Optional: Add points (can get very crowded with many files/points)
    # sns.stripplot(x='File', y='pLDDT', data=df, color="black", size=2, jitter=0.2, alpha=0.3)

    # Add mean value text above each box - use a loop
    ax = plt.gca() # Get current axes
    file_order = df['File'].unique() # Ensure order matches plot

    for i, file_name in enumerate(file_order):
        mean_val = stats_data[file_name]['Mean']
        plt.text(i, 102, f'{mean_val:.1f}', # Position above 100
                 ha='center', # Horizontal alignment
                 va='bottom', # Vertical alignment
                 fontsize=8, 
                 color='red',
                 fontweight='bold')

    plt.title("Distribution of pLDDT Values Across Multiple Structures")
    plt.ylabel("pLDDT Score")
    plt.xlabel("Structure File")
    plt.ylim(0, 108) # Adjust ylim to make space for text
    plt.xticks(rotation=45, ha='right', fontsize=9) # Rotate labels if many files
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calculate_plddt_multi.py <cif_file1> <cif_file2> ...")
        sys.exit(1)

    cif_file_paths = sys.argv[1:] # Get all arguments after the script name
    
    plddt_df, statistics = process_multiple_cifs(cif_file_paths)

    if plddt_df is not None:
        plot_multiple_plddt_distributions(plddt_df, statistics)