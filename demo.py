import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.parser import (
    parse_gamry_file,
    parse_gamry_folder,
    parse_gamry_folder_with_storage,
    GamryFileType
)
from utils.equations import (
    analyze_tafel,
    analyze_eis,
    analyze_galvanic,
    stern_geary_equation,
    validate_tafel_fit
)
from utils.data_storage import GamryDataStorage
'''
This file is used to demo the corrosion analysis.
'''
def plot_tafel(data: pd.DataFrame, result, save_path: str = None):
    """Plot Tafel analysis results."""
    plt.figure(figsize=(10, 6))
    
    # Plot raw data
    plt.semilogx(np.abs(data['Im(A)']), data['Vf(V)'], 'ko', label='Raw data')
    
    # Plot Tafel lines
    E = np.linspace(result.E_corr - 0.3, result.E_corr + 0.3, 100)
    i_a = result.i_corr * np.exp(2.303 * (E - result.E_corr) / result.beta_a)
    i_c = -result.i_corr * np.exp(-2.303 * (E - result.E_corr) / result.beta_c)
    
    plt.semilogx(np.abs(i_a), E, 'r-', label='Anodic Tafel')
    plt.semilogx(np.abs(i_c), E, 'b-', label='Cathodic Tafel')
    
    # Plot E_corr and i_corr
    plt.semilogx(np.abs(result.i_corr), result.E_corr, 'g*', 
                 markersize=15, label='E_corr, i_corr')
    
    plt.xlabel('|i| (A)')
    plt.ylabel('E (V)')
    plt.title('Tafel Plot')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_eis(data: pd.DataFrame, result, save_path: str = None):
    """Plot EIS analysis results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Nyquist plot
    ax1.plot(data['Zreal(Ohm)'], -data['Zimag(Ohm)'], 'ko-')
    ax1.set_xlabel('Z\' (Ω)')
    ax1.set_ylabel('-Z\'\' (Ω)')
    ax1.set_title('Nyquist Plot')
    ax1.grid(True)
    
    # Bode plot
    ax2.semilogx(data['Freq(Hz)'], result['magnitude'], 'ko-', label='|Z|')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('|Z| (Ω)')
    ax2.set_title('Bode Plot')
    ax2.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def process_single_file(file_path: str, file_type: GamryFileType):
    """Process a single Gamry data file."""
    print(f"\nProcessing file: {file_path}")
    
    # Parse the file
    data = parse_gamry_file(file_path, file_type)
    
    if file_type == GamryFileType.POTENTIODYNAMIC:
        # Tafel analysis
        result = analyze_tafel(
            data=data,
            area=1.0,  # cm²
            density=7.8,  # g/cm³
            EW=27.92  # g/eq
        )
        
        # Print results
        print("\nTafel Analysis Results:")
        print(f"E_corr: {result.E_corr:.3f} V")
        print(f"i_corr: {result.i_corr:.2e} A/cm²")
        print(f"Corrosion rate: {result.corrosion_rate:.2f} mm/year")
        print(f"Tafel slopes: βa = {result.beta_a:.3f}, βc = {result.beta_c:.3f} V/decade")
        
        # Validate results
        validations = validate_tafel_fit(data, result)
        print("\nValidation Results:")
        for check, passed in validations.items():
            print(f"{check}: {'✓' if passed else '✗'}")
        
        # Plot results
        plot_tafel(data, result)
        
    elif file_type == GamryFileType.EIS:
        # EIS analysis
        result = analyze_eis(data)
        
        # Print results
        print("\nEIS Analysis Results:")
        print(f"R_s: {result['R_s']:.2f} Ω")
        print(f"R_p: {result['R_p']:.2f} Ω")
        print(f"C: {result['C']:.2e} F")
        
        # Plot results
        plot_eis(data, result)
        
    elif file_type == GamryFileType.GALVANIC:
        # Galvanic analysis
        result = analyze_galvanic(data, area=1.0)
        
        # Print results
        print("\nGalvanic Analysis Results:")
        print(f"Average current: {result['average_current']:.2e} A")
        print(f"Average current density: {result['average_current_density']:.2e} A/cm²")
        print(f"Total charge: {result['total_charge']:.2e} C")

def process_folder(folder_path: str, file_type: GamryFileType):
    """Process all Gamry data files in a folder."""
    print(f"\nProcessing folder: {folder_path}")
    
    # Initialize storage
    storage = GamryDataStorage(base_dir="processed_data")
    
    # Process files and save results
    results = parse_gamry_folder_with_storage(
        folder_path=folder_path,
        file_type=file_type,
        storage=storage,
        recursive=True
    )
    
    print(f"\nProcessed {len(results)} files")
    for file_path, saved_path in results.items():
        print(f"Saved {file_path} to {saved_path}")

def main():
    # Example usage
    data_dir = "raw_data"  # Directory containing your Gamry data files
    
    # Process a single file
    file_path = os.path.join(data_dir, "example_tafel.dta")
    if os.path.exists(file_path):
        process_single_file(file_path, GamryFileType.POTENTIODYNAMIC)
    
    # Process a folder
    if os.path.exists(data_dir):
        process_folder(data_dir, GamryFileType.POTENTIODYNAMIC)

if __name__ == "__main__":
    main()
