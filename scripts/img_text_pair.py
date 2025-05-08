
import os
import pandas as pd
import csv
import openpyxl

"""Match PNGs to Captions from Excel. Make (image,text) pairs"""

def create_image_caption_pairs(excel_path, image_folder, output_csv_path):
    # Load Excel file
    df = pd.read_excel(excel_path, dtype={"Subject ID": str})
    df["Subject ID"] = df["Subject ID"].str.zfill(3)  # Ensure '007' not '7'

    pairs = []

    for file in os.listdir(image_folder):
        if not file.endswith(".png"):
            continue

        # Extract subject ID from filename e.g., BraTS-SSA-00007-000_mip.png → 007
        parts = file.split("-")
        if len(parts) < 3:
            continue
        raw_id = parts[2]  # This gives '00007'
        subject_id = raw_id[-3:]  # Extract last 3 digits → '007'

        # Lookup metadata row
        row = df[df["Subject ID"] == subject_id]

        if not row.empty:
            caption = row.iloc[0]["Primary Tumor"]
            if pd.isna(caption) or str(caption).strip() == "":
                continue  # Skip if caption is missing

            pairs.append({
                "image": file,
                "text": str(caption).strip()
            })

    # Write to CSV
    with open(output_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(pairs)

    print(f"✅ Done. Created {len(pairs)} image-text pairs at: {output_csv_path}")

# === Example Usage ===
excel_path = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/BraTS-Africa_TCIA_datainfo_v2.xlsx"
image_folder = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-t1c_MPI_2D"
output_csv_path = "vlm_image_text_pairs.csv"

create_image_caption_pairs(excel_path, image_folder, output_csv_path)

"""Verification Script"""

def verify_pairs(csv_path, excel_path, image_folder):
    # Load files
    pairs_df = pd.read_csv(csv_path)
    meta_df = pd.read_excel(excel_path, dtype={"Subject ID": str})
    meta_df["Subject ID"] = meta_df["Subject ID"].str.zfill(3)

    for idx, row in pairs_df.iterrows():
        filename = row['image']
        caption = row['text']
        file_id = filename.split("-")[2][-3:]

        meta_row = meta_df[meta_df["Subject ID"] == file_id]

        if meta_row.empty:
            print(f"⚠️ Subject ID {file_id} not found in Excel.")
        elif str(meta_row.iloc[0]["Primary Tumor"]).strip() != caption:
            print(f"❌ Mismatch for {filename}:")
            print(f"  CSV Caption : {caption}")
            print(f"  Excel Value : {meta_row.iloc[0]['Primary Tumor']}")

    print("\n✅ Verification complete.")


verify_pairs("vlm_image_text_pairs.csv", "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/BraTS-Africa_TCIA_datainfo_v2.xlsx", "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-t1c_MPI_2D")

"""File Existence Check for PNGs"""

def check_image_files_exist(csv_path, image_folder):
    df = pd.read_csv(csv_path)
    missing_files = []

    for fname in df['image']:
        full_path = os.path.join(image_folder, fname)
        if not os.path.exists(full_path):
            missing_files.append(fname)

    if missing_files:
        print("❌ The following image files are listed in the CSV but not found in the image folder:")
        for f in missing_files:
            print("   ", f)
    else:
        print("✅ All image files listed in the CSV exist in the folder.")

# === Example Usage ===
csv_path = "vlm_image_text_pairs.csv"
image_folder = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-t1c_MPI_2D"

check_image_files_exist(csv_path, image_folder)

"""Visualize side-by-side comparison (e.g., image + segmentation mask)"""

def display_random_image_text_pairs(csv_path, image_folder, num_samples=5):
    # Load your CSV
    df = pd.read_csv(csv_path)

    # Sample N rows
    sample_df = df.sample(n=min(num_samples, len(df)))

    for _, row in sample_df.iterrows():
        image_path = os.path.join(image_folder, row['image'])
        if os.path.exists(image_path):
            display(IPyImage(filename=image_path))
            print(f"🧠 Caption: {row['text']}\n")
        else:
            print(f"⚠️ Missing file: {image_path}")

# === Example Usage in Jupyter ===
csv_path = "vlm_image_text_pairs.csv"
image_folder = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-t1c_MPI_2D"

display_random_image_text_pairs(csv_path, image_folder, num_samples=5)

"""Visual augmentations to the image pipeline"""

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class VLMImageTextDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None, tokenizer=None):
        self.data = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_folder, row['image'])

        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load text
        text = row['text']
        if self.tokenizer:
            text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        return image, text

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # adjust if RGB
])

# Instantiate dataset
dataset = VLMImageTextDataset(
    csv_path="vlm_image_text_pairs.csv",
    image_folder="/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-t1c_MPI_2D",
    transform=train_transforms
)

# Load a sample
image, text = dataset[0]

from matplotlib import pyplot as plt

for i in range(3):
    img, caption = dataset[i]
    plt.imshow(img.permute(1, 2, 0))
    plt.title(caption)
    plt.axis("off")
    plt.show()

