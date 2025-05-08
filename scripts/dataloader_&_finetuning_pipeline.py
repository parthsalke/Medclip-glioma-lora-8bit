"""
Install Required Packages
"""

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer
import pandas as pd
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

"""Prepare the Dataset + DataLoader"""

class BioMedCLIPDataset(Dataset):
    def __init__(self, excel_path, image_folder, tokenizer, transform=None):
        self.data = pd.read_excel(excel_path)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform

        # Drop rows with missing captions
        self.data = self.data.dropna(subset=['text'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_folder, row["image"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text = row["text"]
        text_inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        return image, text_inputs

"""Split your image-text dataset and organize it"""

# === Inputs ===
excel_path = "vlm_image_text_pairs.xlsx"
image_dir = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-t1c_MPI_2D"
output_dir = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-split"

# === Load Excel file ===
df = pd.read_excel(excel_path)
df = df.dropna(subset=["text"])

# === Split the data ===
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

# === Helper: Copy images to target folder ===
def organize_split(split_df, split_name):
    split_folder = os.path.join(output_dir, split_name)
    images_folder = os.path.join(split_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    records = []
    for _, row in split_df.iterrows():
        filename = row["image"]
        caption = row["text"]
        src = os.path.join(image_dir, filename)
        dst = os.path.join(images_folder, filename)

        if os.path.exists(src):
            shutil.copy(src, dst)
            records.append({"image": filename, "text": caption})
        else:
            print(f"⚠️ Missing image: {filename}")

    # Save captions
    captions_df = pd.DataFrame(records)
    captions_df.to_csv(os.path.join(split_folder, "captions.csv"), index=False)
    print(f"Saved {len(records)} samples to: {split_folder}")

# === Organize train and test folders ===
organize_split(train_df, "train")
organize_split(test_df, "test")

"""Load Train and Test Dataset into a Hugging Face-style Dataset and upload them."""

from datasets import Dataset, DatasetDict, Features, Value, Image
from huggingface_hub import login

# 👇 Authenticate (only once)
login(token="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")

# === Define helper to load one split ===
def load_split(split_name, base_dir):
    split_path = os.path.join(base_dir, split_name)
    csv_path = os.path.join(split_path, "captions.csv")
    img_folder = os.path.join(split_path, "images")

    df = pd.read_csv(csv_path)
    df["image"] = df["image"].apply(lambda x: os.path.join(img_folder, x))

    features = Features({
        "image": Image(),
        "text": Value("string")
    })

    return Dataset.from_pandas(df, features=features)

# === Load train and test ===
train_dataset = load_split("train", base_dir="/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-split")
test_dataset = load_split("test", base_dir="/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-split")

# === Combine into a DatasetDict ===
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# === Upload to HF Hub ===
dataset.push_to_hub("parthsalke/vlm_glioma_dataset")

