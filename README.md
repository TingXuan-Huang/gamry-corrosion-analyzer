# Corrosion Analyzer

A Python-based tool for analyzing electrochemical corrosion data from Gamry potentiostat measurements. This project was developed to assist in the analysis of corrosion behavior of Nickel-Titanium Shape Memory Alloy (NiTi SMA) coupled with steel rebar. (Publication coming soon)

## Project Background

This project was developed during a summer research project focused on understanding the corrosion behavior of NiTi SMA when coupled with steel rebar. The analysis involved various electrochemical techniques including:

- Linear Polarization Resistance (LPR)
- Potentiodynamic Polarization (Tafel)
- Electrochemical Impedance Spectroscopy (EIS)
- Galvanic Corrosion Testing

## Features

- **Comprehensive Data Parsing**
  - Support for multiple Gamry file formats
  - Automatic file type detection
  - Batch processing capabilities
  - Parallel processing for improved performance

- **Advanced Analysis Tools**
  - Tafel analysis with automatic range selection
  - EIS data processing and equivalent circuit fitting
  - Galvanic corrosion analysis
  - LPR calculations with Stern-Geary equation

- **Data Management**
  - Organized storage structure
  - Metadata preservation
  - Easy data retrieval
  - Export capabilities

- **Visualization**
  - Tafel plots
  - Nyquist and Bode plots
  - Galvanic current/voltage plots
  - Customizable plotting options

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TingXuan-Huang/gamry-corrosion-analyzer.git
cd corrosion-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
corrosion-analyzer/
├── utils/
│   ├── parser.py          # Gamry file parsing utilities
│   ├── equations.py       # Electrochemical analysis functions
│   └── data_storage.py    # Data management utilities
├── demo.py               # Example usage script
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Usage

### Basic Usage

1. Place your Gamry data files in the `raw_data` directory:
```bash
mkdir raw_data
# Copy your .dta files here
```

2. Run the demo script:
```bash
python demo.py
```

### Processing Single Files

```python
from utils.parser import parse_gamry_file, GamryFileType
from utils.equations import analyze_tafel

# Parse a Tafel file
data = parse_gamry_file("path/to/file.dta", GamryFileType.POTENTIODYNAMIC)

# Analyze the data
result = analyze_tafel(
    data=data,
    area=1.0,      # cm²
    density=7.8,   # g/cm³
    EW=27.92       # g/eq
)
```

### Batch Processing

```python
from utils.parser import parse_gamry_folder_with_storage
from utils.data_storage import GamryDataStorage

# Initialize storage
storage = GamryDataStorage(base_dir="processed_data")

# Process all files in a folder
results = parse_gamry_folder_with_storage(
    folder_path="raw_data",
    file_type=GamryFileType.POTENTIODYNAMIC,
    storage=storage,
    recursive=True
)
```

## Analysis Methods

### Tafel Analysis
- Automatic range selection for optimal fitting
- Multiple i_corr calculation methods
- Statistical validation
- Confidence interval calculation

### EIS Analysis
- Solution resistance calculation
- Polarization resistance determination
- Capacitance estimation
- Nyquist and Bode plot generation

### Galvanic Corrosion Analysis
- Current density calculation
- Average values and standard deviations
- Total charge calculation
- Time-dependent analysis

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed during a summer research project on NiTi SMA corrosion behavior
- Special thanks to the research team for their guidance and support
- Inspired by the need for efficient electrochemical data analysis tools

## Contact

For questions or suggestions, please open an issue or contact the maintainer. 