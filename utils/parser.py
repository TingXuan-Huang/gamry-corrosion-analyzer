import pandas as pd
from typing import Dict, List, Union, Tuple, Optional
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
from data_storage import GamryDataStorage
'''
This file contains the parser for the Gamry corrosion test.
'''
class GamryFileType(Enum):
    LPR = "lpr"
    GALVANIC = "galvanic"
    POTENTIODYNAMIC = "potentiodynamic"
    EIS = "eis"

@dataclass
class GamryMetadata:
    tag: str
    title: str
    date: str
    time: str
    notes: str

@dataclass
class PotentiostatSettings:
    model: str
    run_time: float
    run_time_unit: str
    sample_time: float
    sample_time_unit: str
    current_limit: float
    current_limit_unit: str
    area: float
    area_unit: str
    density: float
    density_unit: str
    equivalent_weight: float
    equivalent_weight_unit: str
    delay: List[float]
    ir_compensation: str

class GamryParser:
    """Base class for parsing Gamry corrosion test files."""
    
    def __init__(self, file_path: str):
        """
        Initialize the parser with a file path.
        
        Args:
            file_path (str): Path to the Gamry data file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = file_path
        self.data: Dict = {}
        self.metadata: Optional[GamryMetadata] = None
        self.potentiostat_settings: Optional[PotentiostatSettings] = None

    def _read_file(self) -> List[str]:
        """Read the file and return its lines."""
        try:
            with open(self.file_path, 'r') as file:
                return file.readlines()
        except Exception as e:
            raise IOError(f"Error reading file {self.file_path}: {str(e)}")

    def _convert_to_num(self, data: List[str]) -> List[Union[float, str]]:
        """
        Convert strings to numbers where possible.
        
        Args:
            data (List[str]): List of strings to convert
            
        Returns:
            List[Union[float, str]]: List with converted values
        """
        converted = []
        for value in data:
            try:
                converted.append(float(value))
            except ValueError:
                converted.append(value)
        return converted

    def _validate_data(self, data: List[List[Union[float, str]]]) -> bool:
        """
        Validate the parsed data.
        
        Args:
            data (List[List[Union[float, str]]]): Data to validate
            
        Returns:
            bool: True if data is valid
        """
        if not data:
            return False
        # Add more validation as needed
        return True

class LinearPolarizationParser(GamryParser):
    """Parser for Linear Polarization Resistance (LPR) test files."""
    
    def parse(self) -> pd.DataFrame:
        """
        Parse the LPR data file and return a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing the LPR data
        """
        lines = self._read_file()
        data = {"lpr_data": []}
        
        section = None
        for line in lines:
            line = line.strip()
            if line.startswith("CURVE"):
                section = "data"
            elif section == "data" and line.startswith("Pt"):
                continue
            elif section == "data" and line.startswith("#"):
                continue
            elif section == "data" and line:
                row = line.split('\t')
                if len(row) >= 4:  # Ensure we have enough columns
                    used_data = self._convert_to_num(row[:4])
                    data["lpr_data"].append(used_data)
        
        if not self._validate_data(data["lpr_data"]):
            raise ValueError("Invalid or empty data found in file")
            
        df = pd.DataFrame(data["lpr_data"], 
                         columns=["Pt", "Time (s)", "Vf(V)", "Im(A)"])
        return df

class GalvanicParser(GamryParser):
    """Parser for Galvanic Corrosion test files."""
    
    def parse(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Parse the Galvanic Corrosion test file.
        
        Returns:
            Tuple[Dict, pd.DataFrame]: Tuple containing metadata and data DataFrame
        """
        lines = self._read_file()
        data = {
            "metadata": {},
            "potentiostat_settings": {},
            "ocv_curve": [],
            "final_ocv": None,
            "potentiostat_config": {},
            "curve_data_units": [],
            "curve_data": []
        }
        
        section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse metadata
            if line.startswith("TAG"):
                data["metadata"]["TAG"] = line.split('\t')[1]
            elif line.startswith("TITLE"):
                data["metadata"]["TITLE"] = line.split('\t')[2]
            elif line.startswith("DATE"):
                data["metadata"]["DATE"] = line.split('\t')[2]
            elif line.startswith("TIME"):
                data["metadata"]["TIME"] = line.split('\t')[2]
            elif line.startswith("NOTES"):
                data["metadata"]["NOTES"] = line.split('\t')[2]
                
            # Parse potentiostat settings
            elif line.startswith("PSTAT"):
                section = "potentiostat_settings"
                data["potentiostat_settings"]["PSTAT"] = line.split('\t')[2]
                data["potentiostat_settings"]["PSTAT_IDENTIFIER"] = line.split('\t')[1]
            elif section == "potentiostat_settings":
                self._parse_potentiostat_settings(line, data)
                
            # Parse curve data
            elif line.startswith("CURVE"):
                section = "curve_data"
            elif section == "curve_data" and line.startswith("Pt"):
                data["curve_data_units"] = line.split('\t')
            elif section == "curve_data" and line.startswith("#"):
                continue
            elif section == "curve_data" and line:
                self._parse_curve_data(line, data)
        
        if not self._validate_data(data["curve_data"]):
            raise ValueError("Invalid or empty data found in file")
            
        df = pd.DataFrame(data["curve_data"], 
                         columns=["Pt", "Time (s)", "Vf vs Vref (V)", "Im (A)"])
        return data, df

    def _parse_potentiostat_settings(self, line: str, data: Dict) -> None:
        """Parse potentiostat settings from a line."""
        if line.startswith("TRUN"):
            data["potentiostat_settings"]["TRUN"] = line.split('\t')[2]
            data["potentiostat_settings"]["TRUN_UNIT"] = line.split('\t')[3]
        elif line.startswith("SAMPLETIME"):
            data["potentiostat_settings"]["SAMPLETIME"] = line.split('\t')[2]
            data["potentiostat_settings"]["SAMPLETIME_UNIT"] = line.split('\t')[3]
        # Add more settings parsing as needed

    def _parse_curve_data(self, line: str, data: Dict) -> None:
        """Parse curve data from a line."""
        row = line.split('\t')
        if len(row) >= 4:
            try:
                used_data = self._convert_to_num(row[:4])
                data["curve_data"].append(used_data)
            except ValueError as e:
                print(f"Warning: Error converting data: {e}")
                print(f"Problematic data: {row[:4]}")

def parse_gamry_file(file_path: str, file_type: GamryFileType) -> Union[pd.DataFrame, Tuple[Dict, pd.DataFrame]]:
    """
    Parse a Gamry file based on its type.
    
    Args:
        file_path (str): Path to the Gamry file
        file_type (GamryFileType): Type of the Gamry file
        
    Returns:
        Union[pd.DataFrame, Tuple[Dict, pd.DataFrame]]: Parsed data
    """
    if file_type == GamryFileType.LPR:
        parser = LinearPolarizationParser(file_path)
        return parser.parse()
    elif file_type == GamryFileType.GALVANIC:
        parser = GalvanicParser(file_path)
        return parser.parse()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def parse_gamry_folder(folder_path: str, 
                      file_type: GamryFileType,
                      recursive: bool = False,
                      max_workers: int = 4) -> Dict[str, Union[pd.DataFrame, Tuple[Dict, pd.DataFrame]]]:
    """
    Parse all Gamry data files in a folder.
    
    Args:
        folder_path (str): Path to the folder containing Gamry data files
        file_type (GamryFileType): Type of Gamry files to parse
        recursive (bool): Whether to search subdirectories recursively
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        Dict[str, Union[pd.DataFrame, Tuple[Dict, pd.DataFrame]]]: Dictionary mapping file paths to parsed data
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all .dta files in the folder
    if recursive:
        files = list(folder_path.rglob("*.dta"))
    else:
        files = list(folder_path.glob("*.dta"))
    
    if not files:
        logging.warning(f"No .dta files found in {folder_path}")
        return {}
    
    results = {}
    
    def process_file(file_path: Path) -> Tuple[str, Union[pd.DataFrame, Tuple[Dict, pd.DataFrame]]]:
        try:
            data = parse_gamry_file(str(file_path), file_type)
            return str(file_path), data
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return str(file_path), None
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_path, data in executor.map(process_file, files):
            if data is not None:
                results[file_path] = data
    
    return results

def parse_gamry_folder_with_storage(folder_path: str,
                                  file_type: GamryFileType,
                                  storage: GamryDataStorage,
                                  recursive: bool = False,
                                  max_workers: int = 4,
                                  metadata_func: Optional[callable] = None) -> Dict[str, str]:
    """
    Parse all Gamry data files in a folder and save them using the storage system.
    
    Args:
        folder_path (str): Path to the folder containing Gamry data files
        file_type (GamryFileType): Type of Gamry files to parse
        storage (GamryDataStorage): Storage instance to use
        recursive (bool): Whether to search subdirectories recursively
        max_workers (int): Maximum number of parallel workers
        metadata_func (callable): Optional function to generate metadata from file path
        
    Returns:
        Dict[str, str]: Dictionary mapping file paths to saved data paths
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all .dta files in the folder
    if recursive:
        files = list(folder_path.rglob("*.dta"))
    else:
        files = list(folder_path.glob("*.dta"))
    
    if not files:
        logging.warning(f"No .dta files found in {folder_path}")
        return {}
    
    results = {}
    
    def process_file(file_path: Path) -> Tuple[str, str]:
        try:
            # Parse the file
            data = parse_gamry_file(str(file_path), file_type)
            
            # Generate sample ID from file name
            sample_id = file_path.stem
            
            # Generate metadata if function provided
            metadata = None
            if metadata_func:
                metadata = metadata_func(file_path)
            
            # Save the data
            if isinstance(data, tuple):
                data, file_metadata = data
                if metadata:
                    metadata.update(file_metadata)
                else:
                    metadata = file_metadata
            
            saved_path = storage.save_data(
                data=data,
                metadata=metadata,
                test_type=file_type.value,
                sample_id=sample_id
            )
            
            return str(file_path), saved_path
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return str(file_path), None
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_path, saved_path in executor.map(process_file, files):
            if saved_path is not None:
                results[file_path] = saved_path
    
    return results

# Example metadata function
def default_metadata_func(file_path: Path) -> Dict:
    """
    Default function to generate metadata from file path.
    
    Args:
        file_path (Path): Path to the Gamry data file
        
    Returns:
        Dict: Generated metadata
    """
    return {
        "original_file": str(file_path),
        "file_name": file_path.name,
        "file_size": file_path.stat().st_size,
        "last_modified": file_path.stat().st_mtime
    }