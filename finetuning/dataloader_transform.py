import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

# === Load environment variables securely ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "flaviagiammarino/pubmed-clip-vit-base-patch32"
DATASET_ID = "parthsalke/vlm_glioma_dataset"

# === Image transform compatible with CLIP ===
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class GliomaCLIPDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((224, 224))
        return {
            "text": item["text"],
            "image": image
        }

def clip_collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    return processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

def get_model_and_processor():
    model = CLIPModel.from_pretrained(MODEL_ID, token=HF_TOKEN)
    processor = CLIPProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), processor

def get_dataloader(split="train", batch_size=8):
    dataset = load_dataset(DATASET_ID, token=HF_TOKEN)
    processor = CLIPProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    clip_dataset = GliomaCLIPDataset(dataset[split])
    return DataLoader(clip_dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=lambda x: clip_collate_fn(x, processor))

# Optional test block
if __name__ == "__main__":
    model, processor = get_model_and_processor()
    train_loader = get_dataloader()

    batch = next(iter(train_loader))
    print("Input IDs:", batch["input_ids"].shape)
    print("Attention Mask:", batch["attention_mask"].shape)
    print("Pixel Values:", batch["pixel_values"].shape)