import torch
from transformers import pipeline, AutoImageProcessor, Swinv2ForImageClassification

DEVICE = 0 if torch.cuda.is_available() else -1

def load_cataract_model():
    print("🔹 Loading cataract model: amanchandra/cataract-resnet18-classifier")
    return pipeline(
        task="image-classification",
        model="amanchandra/cataract-resnet18-classifier",
        device=DEVICE,
    )

def load_glaucoma_model():
    print("🔹 Loading glaucoma model: pamixsun/swinv2_tiny_for_glaucoma_classification")
    processor = AutoImageProcessor.from_pretrained(
        "pamixsun/swinv2_tiny_for_glaucoma_classification",
        use_fast=True
        )
    model = Swinv2ForImageClassification.from_pretrained("pamixsun/swinv2_tiny_for_glaucoma_classification")
    return processor, model
