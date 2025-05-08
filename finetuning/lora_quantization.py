import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

#Performing FP32 to 8-bit quantization
def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_8bit=True,             # You can switch to 4-bit if desired
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

# Applying LoRA to only Attention (qkv) and MLP (fc1,fc2) layers and freezing all other layers
def get_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],  # Tune based on model architecture
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

def apply_lora(model):
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    return model

def print_lora_layers(model):
    print("\nLoRA Layers:")
    for name, param in model.named_parameters():
        if "lora" in name:
            print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParameter Summary:")
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"% Trainable:          {100 * trainable / total:.2f}%")

# Example usage
if __name__ == "__main__":
    from transformers import CLIPModel

    model_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)

    model = apply_lora(model)
    print_lora_layers(model)
    print_trainable_parameters(model)