import torch
from dataloader_transform import get_dataloader
from lora_quantization import load_lora_clip_model, print_trainable_parameters
from train import train_clip_lora, create_optimizer
from evaluate import evaluate_clip_top1, plot_loss_curve

def main():
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load model with LoRA
    model = load_lora_clip_model()
    print_trainable_parameters(model)

    # 3. Get dataloaders
    train_loader = get_dataloader("train", batch_size=8)
    test_loader = get_dataloader("test", batch_size=8)

    # 4. Optimizer
    optimizer = create_optimizer(model)

    # 5. Train
    loss_history = train_clip_lora(model, train_loader, optimizer, device, epochs=10)

    # 6. Evaluate
    print("Evaluating on train set...")
    evaluate_clip_top1(model, train_loader, device)
    print("Evaluating on test set...")
    evaluate_clip_top1(model, test_loader, device)

    # 7. Visualize loss
    plot_loss_curve(loss_history)

    # 8. Save LoRA-adapted model
    model.save_pretrained("biomedclip_lora_qkv_8bit")
    print("LoRA weights saved to: biomedclip_lora_qkv_8bit")

if __name__ == "__main__":
    main()