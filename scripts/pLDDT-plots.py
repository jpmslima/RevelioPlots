import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import MMCIFParser
import os

def calculate_plddt_stats(cif_file):
    """
    Parses a CIF file to extract pLDDT values and calculates their mean,
    median, and standard deviation.

    Args:
        cif_file (str): The path to the CIF file.

    Returns:
        list: A list of pLDDT values, or None if an error occurs.
    """
    if not os.path.exists(cif_file):
        print(f"Error: File not found at {cif_file}")
        return None
        
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
        plddt_values = []
        
        # Try to get pLDDT from _ma_qa_metric_local first (standard for AlphaFold CIF)
        try:
            plddt_values = [float(row[4]) for row in structure.header['_ma_qa_metric_local']]
            if plddt_values:
                print("Extracted pLDDT from _ma_qa_metric_local.")
                return plddt_values
        except KeyError:
             print("'_ma_qa_metric_local' not found, trying B-factors.")


        # If not found or empty, try B-factors
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Check if it's an amino acid before getting atoms
                    if residue.id[0] == ' ': # Standard residues
                         for atom in residue:
                            # Usually CA (alpha-carbon) holds a representative B-factor
                            if atom.get_name() == 'CA':
                                plddt_values.append(atom.get_bfactor())
                                break # Move to next residue once CA is found

        if not plddt_values:
             print("Could not extract pLDDT values using B-factors either.")
             return None

        print(f"Extracted {len(plddt_values)} pLDDT values from B-factors.")
        return plddt_values

    except Exception as e:
        print(f"Error parsing CIF file: {e}")
        return None

def plot_plddt_distribution(plddt_values, cif_file_name):
    """
    Generates a box plot of pLDDT values with individual data points
    and includes statistics on the plot.

    Args:
        plddt_values (list): A list of pLDDT values.
        cif_file_name (str): Name of the input CIF file for the title.
    """
    if not plddt_values:
        print("No pLDDT values to plot.")
        return

    mean_plddt = np.mean(plddt_values)
    median_plddt = np.median(plddt_values)
    std_dev_plddt = np.std(plddt_values)

    print(f"\n--- Statistics for {cif_file_name} ---")
    print(f"Mean pLDDT: {mean_plddt:.2f}")
    print(f"Median pLDDT: {median_plddt:.2f}")
    print(f"Standard Deviation of pLDDT: {std_dev_plddt:.2f}")
    print("--------------------------------------\n")


    plt.figure(figsize=(8, 7)) # Increased height slightly for text
    
    # Create the plot
    ax = sns.boxplot(y=plddt_values, color="lightblue", width=0.4)
    sns.stripplot(y=plddt_values, color="black", size=2.5, jitter=0.2, alpha=0.6)

    # Prepare the text for statistics
    stats_text = (
        f"Mean: {mean_plddt:.2f}\n"
        f"Median: {median_plddt:.2f}\n"
        f"Std Dev: {std_dev_plddt:.2f}"
    )

    # Add the text to the plot
    # We use `transform=ax.transAxes` for positioning relative to the axes
    # (0,0) is bottom-left, (1,1) is top-right.
    # We add a box around the text for better readability.
    plt.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # Set title and labels
    plt.title(f"Distribution of pLDDT Values for {cif_file_name}")
    plt.ylabel("pLDDT Score")
    plt.xlabel("Structure") # Added x-label for context
    plt.ylim(0, 100) # pLDDT values range from 0 to 100
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove x-axis ticks as there's only one category
    plt.xticks([]) 
    
    plt.tight_layout() # Adjust layout
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_plddt.py <cif_file>")
        sys.exit(1)

    cif_file_path = sys.argv[1]
    file_name = os.path.basename(cif_file_path) # Get just the filename
    plddts = calculate_plddt_stats(cif_file_path)

    if plddts:
        plot_plddt_distribution(plddts, file_name)