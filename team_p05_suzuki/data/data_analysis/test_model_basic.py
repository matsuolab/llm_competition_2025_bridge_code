import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def test_model():
    print_flush("=== Testing Model Loading ===")

    # Check Python version
    print_flush(f"Python version: {sys.version}")

    # Check PyTorch version
    print_flush(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    print_flush(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print_flush(f"CUDA device: {torch.cuda.get_device_name()}")
        print_flush(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Model configuration - using a smaller model for testing
    model_id = "bert-base-uncased"  # Much smaller model for initial testing

    try:
        print_flush(f"Loading tokenizer from {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        print_flush("Tokenizer loaded successfully!")

        print_flush(f"Loading model from {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, load_in_4bit=True, trust_remote_code=True
        )
        print_flush("Model loaded successfully!")

        # Test text
        text = "What is machine learning?"
        print_flush(f"Tokenizing test text: '{text}'")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        )
        print_flush(f"Input shape: {inputs['input_ids'].shape}")

        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print_flush("Generating embeddings...")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        print_flush(f"Hidden states shape: {hidden_states.shape}")
        print_flush("Test completed successfully!")

        return True

    except Exception as e:
        print_flush(f"Error during test: {str(e)}")
        import traceback

        print_flush(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
