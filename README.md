# MedCLIP Glioma Classification Pipeline

Approximately half of all primary brain tumors are glial cell neoplasms (Gliomas), and more than three quarters of all gliomas are astrocytoma.

This repository implements an end-to-end pipeline for **preprocessing, fine-tuning, and evaluating a medical Vision-Language Model (VLM)** on Glioma MRI scans. It fine-tunes pretrained [PubMedCLIP](https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32) model using Supervised Fine-tuning using **LoRA adapters**, **8-bit quantization**, and supervised contrastive learning, and is fully modular and container-ready, [MedCLIP](https://huggingface.co/parthsalke/medclip-glioma-lora-8bit)

The project is organized into two main modules: `finetuning` and `scripts`, each with their own main execution file.

1.	**Dataset**- Cross-Embodiment dataset from BRATS [TCIA](https://www.cancerimagingarchive.net/) dataset which have T1, T1-CE, T2 and T2-FLAIR and then converted DICOM & NIFTI format to PNG format, as input to VLM required is 2D PNG, converted 3D NIFTI format to 2D PNG using [MIP](https://en.wikipedia.org/wiki/Maximum_intensity_projection#:~:text=In%20scientific%20visualization%2C%20a%20maximum,to%20the%20plane%20of%20projection.) technique along axial & coronal plane.

2.	**Technique-** Contrastive Learning was best for the custom Glioma VLM project because it aligns image and text embeddings in a shared space, allowing the model to learn visual-textual associations without needing explicit class labels.
3.	**Fine-tuning**- Froze the main model weights and trained only the lightweight LoRA adapters, specifically applied to the Attention (q,k,v) and MLP (fc) layers in both the vision encoder and language encoder. This setup enables parameter-efficient fine-tuning, with only **~1.2%** of the total model parameters being updated.
4.	**Training-** using [Hyperbolic](https://app.hyperbolic.xyz/) / [RunPod](https://www.runpod.io/)
5.	**Evaluation-** Zero-shot and fine-tuned model capabilities across two cross-modal retrieval tasks: Zero-shot Image-to-Text Retrieval and Zero-shot Text-to-Image Retrieval.
   Both tasks are evaluated using Recall@K (R@1, R@5, R@10) metrics. Additionally, we evaluated the model on a fine-tuned classification task and evaluated with other similar medical/brain pre-trained models for top-k accuracy and AUC score.


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

To run the `finetuning` pipeline:
```bash
python finetuning/main.py
```

### Running Scripts

To execute the main script in the `scripts` folder:
```bash
python scripts/main.py
```

### Evaluation
| Model                           | Zero-shot Top-5% accuracy           | AUC (%) |
|--------------------------------|--------------------------------------|---------|
| **MedCLIP (8-bit)**            | **74%**                              |**76.2** |
| MedCLIP (Vanilla)              | 64%                                  | 54.9    |
| PubMedCLIP (Finetuned)         | 45%                                  | 54.7    |
| Llava Med_v1.5                 | 71%                                  | 62.3    |
| OpenAI- CLIP (Vanilla)         | 57%                                  | 45.3    |


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
