import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# === Visualization ===
def plot_loss_curve(loss_history, title="Training Loss Over Epochs", save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Average Contrastive Loss")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Saved loss plot to {save_path}")
    plt.show()

# === Evaluation: Zero-shot Top-1 Accuracy ===
def evaluate_clip_top1(model, dataloader, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            image_embeds = F.normalize(model.get_image_features(pixel_values=pixel_values), dim=-1)
            text_embeds = F.normalize(model.get_text_features(input_ids=input_ids, attention_mask=attention_mask), dim=-1)

            similarity = image_embeds @ text_embeds.T  # [B x B]
            preds = similarity.argmax(dim=1)
            labels = torch.arange(similarity.size(0), device=device)

            correct += (preds == labels).sum().item()
            total += similarity.size(0)

    accuracy = correct / total
    print(f"🎯 Zero-shot Top-1 Accuracy: {accuracy:.2%}")
    return accuracy

# === Optional: Run from script ===
if __name__ == "__main__":
    from dataloader_transform import GliomaCLIPDataset, clip_collate_fn
    from datasets import load_dataset
    import os

    HF_TOKEN = os.getenv("HF_TOKEN")
    dataset = load_dataset("parthsalke/vlm_glioma_dataset", token=HF_TOKEN)
    test_data = dataset["test"]

    test_dataset = GliomaCLIPDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=clip_collate_fn)

    from lora_quantization import apply_lora
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = apply_lora(model)

    accuracy = evaluate_clip_top1(model, test_loader, device)

    # Save the LoRA-adapted model
    model.save_pretrained("biomedclip_lora_qkv_8bit")
    print("✅ Model saved at: biomedclip_lora_qkv_8bit")