import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_ID = "BAAI/bge-small-en-v1.5"
EXPORT_DIR = "onnx_model"

def export_model():
    print(f"Exporting {MODEL_ID} to ONNX...")
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
    
    # Save the ONNX model and tokenizer
    os.makedirs(EXPORT_DIR, exist_ok=True)
    model.save_pretrained(EXPORT_DIR)
    tokenizer.save_pretrained(EXPORT_DIR)
    print(f"Model exported successfully to {EXPORT_DIR}")

if __name__ == "__main__":
    export_model()
