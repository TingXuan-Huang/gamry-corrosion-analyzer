import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional, Union, List
import warnings
from dataclasses import dataclass
'''
This file contains the equations for the corrosion analysis.
'''
def stern_geary_equation(lpr_data, beta_a = 1.2 * 1e-2, beta_c = 1.2 * 1e-2, range = 0.0051):
    """
    Calculate corrosion current density using Stern-Geary equation for LPR data.
    
    Args:
        lpr_data: DataFrame containing LPR data
        beta_a: Anodic Tafel slope (V/decade)
        beta_c: Cathodic Tafel slope (V/decade)
        range: Potential range around E_corr for linear fit
        
    Returns:
        Tuple of (i_corr, R_p, B, E_corr)
    """
    # Calculate the R_p
    R_p, E_corr = cal_Rp(lpr_data, range=range)

    # Calculate the Stern-Geary constant B
    B = (beta_a * beta_c) / (2.303 * (beta_a + beta_c))
    
    return B / R_p, R_p, B, E_corr

def calc_corrosion_rate(i_corr, area, density, EW, K = 1.288 * 1e5):
    """
    Calculate corrosion rate from corrosion current density.
    
    Args:
        i_corr: Corrosion current density (A/cm²)
        area: Sample area (cm²)
        density: Material density (g/cm³)
        EW: Equivalent weight (g/eq)
        K: Conversion factor (default: 1.288e5 for mm/year)
        
    Returns:
        Corrosion rate in mm/year
    """
    return (i_corr * EW * K) / (area * density)

def cal_Rp(lpr_data: pd.DataFrame, range = 0.0051):
    """
    Calculate polarization resistance from LPR data.
    
    Args:
        lpr_data: DataFrame containing LPR data
        range: Potential range around E_corr for linear fit
        
    Returns:
        Tuple of (R_p, E_corr)
    """
    E_corr = find_E_corr(lpr_data)
    lpr_data['In Range'] = (lpr_data['Vf(V)'] >= (E_corr - range)) & (lpr_data['Vf(V)'] <= (E_corr + range))
    eq_data = lpr_data[lpr_data['In Range'] == True].copy()
    
    slope, intercept, r_value, p_value, std_err = linregress(np.array(eq_data['Im(A)']), np.array(eq_data['Vf(V)']))
    R_p = slope
    
    return R_p, E_corr

def find_E_corr(lpr_data):
    """Find corrosion potential from LPR data."""
    min_idx = abs(lpr_data['Im(A)']).idxmin()
    return lpr_data['Vf(V)'][min_idx]

# New functions for other electrochemical techniques

@dataclass
class TafelFitResult:
    """Class to store Tafel analysis results."""
    i_corr: float
    E_corr: float
    beta_a: float
    beta_c: float
    r2_a: float
    r2_c: float
    fit_range_a: Tuple[float, float]
    fit_range_c: Tuple[float, float]
    corrosion_rate: float
    confidence_interval: Tuple[float, float]

def analyze_tafel(data: pd.DataFrame,
                 area: float,
                 density: float,
                 EW: float,
                 min_points: int = 5,
                 r2_threshold: float = 0.95,
                 potential_range: float = 0.25) -> TafelFitResult:
    """
    Comprehensive Tafel analysis with automatic range selection and validation.
    
    Args:
        data: DataFrame with columns ['Vf(V)', 'Im(A)']
        area: Sample area (cm²)
        density: Material density (g/cm²)
        EW: Equivalent weight (g/eq)
        min_points: Minimum number of points for fitting
        r2_threshold: Minimum R² value for acceptable fit
        potential_range: Maximum potential range from E_corr for fitting
        
    Returns:
        TafelFitResult object containing analysis results
    """
    # Find E_corr
    E_corr = find_E_corr(data)
    
    # Separate anodic and cathodic branches
    anodic = data[data['Vf(V)'] > E_corr].copy()
    cathodic = data[data['Vf(V)'] < E_corr].copy()
    
    # Find optimal fitting ranges
    fit_range_a = find_optimal_tafel_range(anodic, E_corr, potential_range, min_points, r2_threshold)
    fit_range_c = find_optimal_tafel_range(cathodic, E_corr, -potential_range, min_points, r2_threshold)
    
    # Fit Tafel lines
    beta_a, r2_a = fit_tafel_line(anodic, fit_range_a)
    beta_c, r2_c = fit_tafel_line(cathodic, fit_range_c)
    
    # Calculate i_corr using both methods
    i_corr_intersection = calculate_i_corr_intersection(beta_a, beta_c, E_corr, anodic, cathodic)
    i_corr_extrapolation = calculate_i_corr_extrapolation(beta_a, beta_c, E_corr, anodic, cathodic)
    
    # Use the more reliable method
    i_corr = select_better_i_corr(i_corr_intersection, i_corr_extrapolation, r2_a, r2_c)
    
    # Calculate corrosion rate
    corr_rate = calc_corrosion_rate(i_corr, area, density, EW)
    
    # Calculate confidence interval
    ci = calculate_confidence_interval(i_corr, r2_a, r2_c, len(anodic), len(cathodic))
    
    return TafelFitResult(
        i_corr=i_corr,
        E_corr=E_corr,
        beta_a=beta_a,
        beta_c=beta_c,
        r2_a=r2_a,
        r2_c=r2_c,
        fit_range_a=fit_range_a,
        fit_range_c=fit_range_c,
        corrosion_rate=corr_rate,
        confidence_interval=ci
    )

def find_optimal_tafel_range(data: pd.DataFrame,
                            E_corr: float,
                            max_range: float,
                            min_points: int,
                            r2_threshold: float) -> Tuple[float, float]:
    """
    Find the optimal potential range for Tafel fitting.
    
    Args:
        data: DataFrame with columns ['Vf(V)', 'Im(A)']
        E_corr: Corrosion potential
        max_range: Maximum potential range from E_corr
        min_points: Minimum number of points for fitting
        r2_threshold: Minimum R² value for acceptable fit
        
    Returns:
        Tuple of (start_potential, end_potential)
    """
    best_r2 = 0
    best_range = (0, 0)
    
    # Try different potential ranges
    for start in np.linspace(0, max_range, 20):
        end = start + 0.1  # 100 mV range
        if end > max_range:
            break
            
        # Get data in range
        mask = (data['Vf(V)'] >= E_corr + start) & (data['Vf(V)'] <= E_corr + end)
        range_data = data[mask]
        
        if len(range_data) < min_points:
            continue
            
        # Try fitting
        try:
            slope, _, r2, _, _ = linregress(
                np.log10(np.abs(range_data['Im(A)'])),
                range_data['Vf(V)']
            )
            
            if r2 > best_r2 and r2 >= r2_threshold:
                best_r2 = r2
                best_range = (E_corr + start, E_corr + end)
        except:
            continue
    
    return best_range

def fit_tafel_line(data: pd.DataFrame,
                  potential_range: Tuple[float, float]) -> Tuple[float, float]:
    """
    Fit Tafel line to data within potential range.
    
    Args:
        data: DataFrame with columns ['Vf(V)', 'Im(A)']
        potential_range: Tuple of (start_potential, end_potential)
        
    Returns:
        Tuple of (beta, r2)
    """
    mask = (data['Vf(V)'] >= potential_range[0]) & (data['Vf(V)'] <= potential_range[1])
    range_data = data[mask]
    
    slope, _, r2, _, _ = linregress(
        np.log10(np.abs(range_data['Im(A)'])),
        range_data['Vf(V)']
    )
    
    return abs(slope), r2

def calculate_i_corr_intersection(beta_a: float,
                                beta_c: float,
                                E_corr: float,
                                anodic: pd.DataFrame,
                                cathodic: pd.DataFrame) -> float:
    """Calculate i_corr using intersection method."""
    def tafel_line(E, beta, i0):
        return i0 * np.exp(2.303 * (E - E_corr) / beta)
    
    # Initial guess
    i_corr_guess = np.min([
        np.abs(anodic['Im(A)'].iloc[0]),
        np.abs(cathodic['Im(A)'].iloc[0])
    ])
    
    # Fit both branches
    popt_a, _ = curve_fit(
        lambda E, i0: tafel_line(E, beta_a, i0),
        anodic['Vf(V)'],
        anodic['Im(A)'],
        p0=[i_corr_guess]
    )
    
    popt_c, _ = curve_fit(
        lambda E, i0: tafel_line(E, beta_c, i0),
        cathodic['Vf(V)'],
        cathodic['Im(A)'],
        p0=[i_corr_guess]
    )
    
    return np.mean([popt_a[0], popt_c[0]])

def calculate_i_corr_extrapolation(beta_a: float,
                                 beta_c: float,
                                 E_corr: float,
                                 anodic: pd.DataFrame,
                                 cathodic: pd.DataFrame) -> float:
    """Calculate i_corr using extrapolation method."""
    # Get points closest to E_corr
    anodic_point = anodic.iloc[0]
    cathodic_point = cathodic.iloc[0]
    
    # Calculate i_corr using both branches
    i_corr_a = anodic_point['Im(A)'] * np.exp(-2.303 * (anodic_point['Vf(V)'] - E_corr) / beta_a)
    i_corr_c = cathodic_point['Im(A)'] * np.exp(-2.303 * (cathodic_point['Vf(V)'] - E_corr) / beta_c)
    
    return np.mean([i_corr_a, i_corr_c])

def select_better_i_corr(i_corr_intersection: float,
                        i_corr_extrapolation: float,
                        r2_a: float,
                        r2_c: float) -> float:
    """Select the more reliable i_corr value."""
    # If both fits are good, use intersection method
    if r2_a > 0.95 and r2_c > 0.95:
        return i_corr_intersection
    
    # If one fit is poor, use extrapolation method
    return i_corr_extrapolation

def calculate_confidence_interval(i_corr: float,
                                r2_a: float,
                                r2_c: float,
                                n_a: int,
                                n_c: int,
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for i_corr.
    
    Args:
        i_corr: Corrosion current density
        r2_a: R² value for anodic branch
        r2_c: R² value for cathodic branch
        n_a: Number of points in anodic branch
        n_c: Number of points in cathodic branch
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Calculate standard error
    se_a = np.sqrt((1 - r2_a) / (n_a - 2))
    se_c = np.sqrt((1 - r2_c) / (n_c - 2))
    
    # Combined standard error
    se = np.sqrt(se_a**2 + se_c**2)
    
    # Calculate confidence interval
    z = 1.96  # 95% confidence
    ci = z * se * i_corr
    
    return (i_corr - ci, i_corr + ci)

def validate_tafel_fit(data: pd.DataFrame,
                      result: TafelFitResult) -> Dict[str, bool]:
    """
    Validate Tafel analysis results.
    
    Args:
        data: Original data
        result: TafelFitResult object
        
    Returns:
        Dictionary of validation results
    """
    validations = {
        'beta_a_range': 0.05 <= result.beta_a <= 0.2,  # Typical range for beta_a
        'beta_c_range': 0.05 <= result.beta_c <= 0.2,  # Typical range for beta_c
        'r2_threshold': result.r2_a >= 0.95 and result.r2_c >= 0.95,
        'i_corr_positive': result.i_corr > 0,
        'confidence_interval': result.confidence_interval[0] > 0
    }
    
    # Check for mass transport effects
    anodic = data[data['Vf(V)'] > result.E_corr]
    cathodic = data[data['Vf(V)'] < result.E_corr]
    
    # Check for limiting current in cathodic branch
    if len(cathodic) > 0:
        i_lim = np.max(np.abs(cathodic['Im(A)']))
        validations['no_limiting_current'] = i_lim < 10 * result.i_corr
    
    return validations

def analyze_eis(data: pd.DataFrame, 
               frequency_col: str = 'Freq(Hz)',
               z_real_col: str = 'Zreal(Ohm)',
               z_imag_col: str = 'Zimag(Ohm)') -> Dict:
    """
    Analyze Electrochemical Impedance Spectroscopy (EIS) data.
    
    Args:
        data: DataFrame containing EIS data
        frequency_col: Column name for frequency
        z_real_col: Column name for real impedance
        z_imag_col: Column name for imaginary impedance
        
    Returns:
        Dictionary containing analysis results
    """
    # Calculate magnitude and phase
    Z = data[z_real_col] + 1j * data[z_imag_col]
    magnitude = np.abs(Z)
    phase = np.angle(Z, deg=True)
    
    # Find high frequency intercept (solution resistance)
    R_s = data[z_real_col].iloc[0]
    
    # Find low frequency intercept (total resistance)
    R_t = data[z_real_col].iloc[-1]
    
    # Calculate polarization resistance
    R_p = R_t - R_s
    
    # Find frequency at maximum phase angle
    max_phase_idx = np.argmax(np.abs(phase))
    f_max = data[frequency_col].iloc[max_phase_idx]
    
    # Calculate capacitance
    C = 1 / (2 * np.pi * f_max * R_p)
    
    return {
        'R_s': R_s,
        'R_p': R_p,
        'R_t': R_t,
        'C': C,
        'f_max': f_max,
        'magnitude': magnitude,
        'phase': phase
    }

def analyze_galvanic(data: pd.DataFrame, 
                    area: float,
                    time_col: str = 'Time (s)',
                    current_col: str = 'Im (A)',
                    voltage_col: str = 'Vf vs Vref (V)') -> Dict:
    """
    Analyze galvanic corrosion data.
    
    Args:
        data: DataFrame containing galvanic data
        area: Sample area (cm²)
        time_col: Column name for time
        current_col: Column name for current
        voltage_col: Column name for voltage
        
    Returns:
        Dictionary containing analysis results
    """
    # Calculate current density
    current_density = data[current_col] / area
    
    # Calculate average values
    avg_current = np.mean(data[current_col])
    avg_current_density = np.mean(current_density)
    avg_voltage = np.mean(data[voltage_col])
    
    # Calculate standard deviations
    std_current = np.std(data[current_col])
    std_voltage = np.std(data[voltage_col])
    
    # Calculate total charge
    total_charge = np.trapz(data[current_col], data[time_col])
    
    return {
        'average_current': avg_current,
        'average_current_density': avg_current_density,
        'average_voltage': avg_voltage,
        'current_std': std_current,
        'voltage_std': std_voltage,
        'total_charge': total_charge
    }

def calculate_icorr_from_eis(R_p: float, 
                           beta_a: float = 0.12,
                           beta_c: float = 0.12) -> float:
    """
    Calculate corrosion current density from EIS data using Stern-Geary equation.
    
    Args:
        R_p: Polarization resistance (Ohm·cm²)
        beta_a: Anodic Tafel slope (V/decade)
        beta_c: Cathodic Tafel slope (V/decade)
        
    Returns:
        Corrosion current density (A/cm²)
    """
    B = (beta_a * beta_c) / (2.303 * (beta_a + beta_c))
    return B / R_p