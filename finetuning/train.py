import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

def train_clip_lora(model, dataloader, optimizer, device, epochs=1):
    """
    Fine-tunes a CLIP-based model using LoRA adapters.

    Args:
        model: A CLIPModel with LoRA layers applied.
        dataloader: PyTorch DataLoader containing image-text batches.
        optimizer: Optimizer (e.g., AdamW).
        device: torch.device("cuda" or "cpu").
        epochs: Number of training epochs.

    Returns:
        List of average training loss per epoch.
    """
    model.train()
    model.to(device)
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            image_embeds = F.normalize(model.get_image_features(pixel_values=pixel_values), dim=-1)
            text_embeds = F.normalize(model.get_text_features(input_ids=input_ids, attention_mask=attention_mask), dim=-1)

            # Bidirectional contrastive loss
            logits = image_embeds @ text_embeds.T
            labels = torch.arange(logits.size(0), device=device)

            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    return loss_history


def create_optimizer(model, lr=2e-4):
    """
    Creates an AdamW optimizer for the model.

    Args:
        model: PyTorch model.
        lr: Learning rate.

    Returns:
        torch.optim.AdamW instance.
    """
    return optim.AdamW(model.parameters(), lr=lr)


# Example usage block (optional)
if __name__ == "__main__":
    from dataloader_transform import get_dataloader, get_model_and_processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = get_model_and_processor()
    train_loader = get_dataloader()
    optimizer = create_optimizer(model)

    history = train_clip_lora(model, train_loader, optimizer, device, epochs=10)