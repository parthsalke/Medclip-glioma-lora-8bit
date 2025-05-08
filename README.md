# MRI Data Analysis for Gliomas Tumors

This repository contains code for analyzing MRI data related to gliomas tumors using deep learning techniques. The project is organized into two main modules: `finetuning` and `scripts`, each with their own main execution file.

## Requirements

- Python 3.x
- See `requirements.txt` for all dependencies

## Installation

1. **Clone this repository:**
   ```bash
   git clone [your-repository-url]
   cd [repository-name]
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── finetuning/
│   ├── file1.py
│   ├── file2.py
│   ├── file3.py
│   ├── file4.py
│   └── main.py
├── scripts/
│   ├── script1.py
│   ├── script2.py
│   ├── script3.py
│   ├── script4.py
│   └── main.py
├── requirements.txt
└── ...
```

- **finetuning/**: Contains scripts for model fine-tuning.
- **scripts/**: Contains utility and data processing scripts.

## Usage

### Running Fine-tuning

To run the fine-tuning pipeline:
```bash
python finetuning/main.py
```

### Running Scripts

To execute the main script in the `scripts` folder:
```bash
python scripts/main.py
```

## Features

- MRI data processing and analysis
- Deep learning model fine-tuning
- Tumor detection and classification
- Data visualization

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

[Add your license information here]

## Contact

[Add your contact information here] 