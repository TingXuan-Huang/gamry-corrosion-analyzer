import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Union, Optional
from pathlib import Path
import h5py
import numpy as np
'''
This class is used to store and retrieve data from the Gamry corrosion test.
'''
class GamryDataStorage:
    """Handles storage and retrieval of Gamry corrosion test data."""
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the data storage with a base directory.
        
        Args:
            base_dir (str): Base directory for storing data
        """
        self.base_dir = Path(base_dir)
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create the necessary directory structure."""
        # Main directories
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create directories if they don't exist
        for directory in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories for different test types
        for test_type in ["lpr", "galvanic", "eis", "potentiodynamic"]:
            (self.processed_dir / test_type).mkdir(exist_ok=True)
            (self.metadata_dir / test_type).mkdir(exist_ok=True)
    
    def save_data(self, 
                 data: Union[pd.DataFrame, Dict],
                 metadata: Optional[Dict] = None,
                 test_type: str = "lpr",
                 sample_id: str = None,
                 timestamp: Optional[str] = None) -> str:
        """
        Save the parsed data in a structured format.
        
        Args:
            data: DataFrame or dictionary containing the test data
            metadata: Dictionary containing test metadata
            test_type: Type of test (lpr, galvanic, eis, potentiodynamic)
            sample_id: Unique identifier for the sample
            timestamp: Optional timestamp for the test
            
        Returns:
            str: Path to the saved data
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        if sample_id is None:
            sample_id = f"sample_{timestamp}"
            
        # Create test-specific directory
        test_dir = self.processed_dir / test_type / sample_id
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data in HDF5 format
        data_path = test_dir / f"{timestamp}_data.h5"
        if isinstance(data, pd.DataFrame):
            data.to_hdf(data_path, key='data', mode='w')
        else:
            with h5py.File(data_path, 'w') as f:
                for key, value in data.items():
                    if isinstance(value, (np.ndarray, list)):
                        f.create_dataset(key, data=value)
                    else:
                        f.attrs[key] = value
        
        # Save metadata in JSON format
        if metadata:
            metadata_path = self.metadata_dir / test_type / f"{sample_id}_{timestamp}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        return str(data_path)
    
    def load_data(self, 
                 test_type: str,
                 sample_id: str,
                 timestamp: Optional[str] = None) -> tuple:
        """
        Load data for a specific test.
        
        Args:
            test_type: Type of test
            sample_id: Sample identifier
            timestamp: Optional timestamp for specific test
            
        Returns:
            tuple: (data, metadata)
        """
        test_dir = self.processed_dir / test_type / sample_id
        
        if timestamp:
            data_path = test_dir / f"{timestamp}_data.h5"
            metadata_path = self.metadata_dir / test_type / f"{sample_id}_{timestamp}_metadata.json"
        else:
            # Get the most recent data if timestamp not specified
            data_files = list(test_dir.glob("*_data.h5"))
            if not data_files:
                raise FileNotFoundError(f"No data found for {test_type}/{sample_id}")
            data_path = max(data_files, key=lambda x: x.stat().st_mtime)
            timestamp = data_path.stem.split('_')[0]
            metadata_path = self.metadata_dir / test_type / f"{sample_id}_{timestamp}_metadata.json"
        
        # Load data
        try:
            data = pd.read_hdf(data_path)
        except:
            with h5py.File(data_path, 'r') as f:
                data = {key: f[key][:] for key in f.keys()}
                data.update(f.attrs)
        
        # Load metadata
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return data, metadata
    
    def list_samples(self, test_type: Optional[str] = None) -> Dict[str, list]:
        """
        List all available samples and their timestamps.
        
        Args:
            test_type: Optional test type to filter by
            
        Returns:
            Dict[str, list]: Dictionary mapping sample IDs to their timestamps
        """
        samples = {}
        
        if test_type:
            test_types = [test_type]
        else:
            test_types = ["lpr", "galvanic", "eis", "potentiodynamic"]
        
        for tt in test_types:
            test_dir = self.processed_dir / tt
            if not test_dir.exists():
                continue
                
            for sample_dir in test_dir.iterdir():
                if sample_dir.is_dir():
                    timestamps = []
                    for data_file in sample_dir.glob("*_data.h5"):
                        timestamp = data_file.stem.split('_')[0]
                        timestamps.append(timestamp)
                    if timestamps:
                        samples[f"{tt}/{sample_dir.name}"] = sorted(timestamps)
        
        return samples
    
    def export_to_excel(self,
                       test_type: str,
                       sample_id: str,
                       timestamp: Optional[str] = None,
                       output_path: Optional[str] = None) -> str:
        """
        Export data to Excel format.
        
        Args:
            test_type: Type of test
            sample_id: Sample identifier
            timestamp: Optional timestamp for specific test
            output_path: Optional custom output path
            
        Returns:
            str: Path to the exported Excel file
        """
        data, metadata = self.load_data(test_type, sample_id, timestamp)
        
        if output_path is None:
            output_path = self.processed_dir / test_type / sample_id / f"export_{timestamp or 'latest'}.xlsx"
        
        with pd.ExcelWriter(output_path) as writer:
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name='Data', index=False)
            else:
                for key, value in data.items():
                    if isinstance(value, (np.ndarray, list)):
                        pd.DataFrame(value).to_excel(writer, sheet_name=key, index=False)
            
            if metadata:
                pd.DataFrame([metadata]).to_excel(writer, sheet_name='Metadata', index=False)
        
        return str(output_path) 