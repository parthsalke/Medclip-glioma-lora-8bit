# MRI Data Analysis for Gliomas Tumors

This repository contains code for analyzing MRI data related to gliomas tumors using deep learning techniques. The project is organized into two main modules: `finetuning` and `scripts`, each with their own main execution file.

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/parthsalke/Medclip-glioma-lora-8bit
   cd Medclip-glioma-lora-8bit
   ```

2. **(Recommended) Create a virtual environment:**
   
   On Mac:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```
   On Windows: 
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── finetuning/
│   ├── dataloader_transform.py
│   ├── evaluate.py
│   ├── lora_quantization.py
│   ├── train.py
│   └── main.py
├── scripts/
│   ├── dataloader_&_finetuning_pipeline.py
│   ├── extract_t1c_files_1modality.py
│   ├── img_text_pair.py
│   ├── nifti3d_to_2dpng.py
│   └── main.py
├── requirements.txt
└── README.md
```

- **finetuning/**: Contains scripts for model fine-tuning.
- **scripts/**: Contains data extraction, utility and data processing scripts.

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
