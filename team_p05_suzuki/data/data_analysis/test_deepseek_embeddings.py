import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("Starting DeepSeek model test...", flush=True)

    # Basic PyTorch test
    print("\nTesting PyTorch:", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}", flush=True)
        print(f"Initial CUDA memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", flush=True)

    # Model configuration
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    print(f"\nTesting with model: {model_id}", flush=True)

    try:
        # Load tokenizer
        print("\nLoading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        print("Tokenizer loaded successfully!", flush=True)

        # Load model with 4-bit quantization
        print("\nLoading model with 4-bit quantization...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, load_in_4bit=True, trust_remote_code=True
        )
        print("Model loaded successfully!", flush=True)

        if torch.cuda.is_available():
            print(f"CUDA memory after model load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", flush=True)

        # Test texts
        texts = ["What is machine learning?", "How does photosynthesis work?", "Explain quantum mechanics."]

        # Get embeddings for each text
        print("\nGenerating embeddings for test texts...", flush=True)
        all_embeddings = []

        for text in texts:
            print(f"\nProcessing: {text}", flush=True)

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
            ).to(model.device)

            print(f"Input shape: {inputs['input_ids'].shape}", flush=True)

            # Get hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

                # Apply attention mask and average
                mask = inputs["attention_mask"].unsqueeze(-1)
                hs = hidden_states * mask
                summed = hs.sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                emb = summed / lengths

                # Apply layer normalization if available
                if hasattr(model, "model") and hasattr(model.model, "final_layernorm"):
                    emb = model.model.final_layernorm(emb)
                elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                    emb = model.transformer.ln_f(emb)

                # Normalize embeddings
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

                # Convert to numpy and store
                emb_np = emb.float().cpu().numpy()
                all_embeddings.append(emb_np)

                print(f"Embedding shape: {emb_np.shape}", flush=True)

        print("\nAll embeddings generated successfully!", flush=True)
        print(f"Number of embeddings: {len(all_embeddings)}", flush=True)
        print(f"Embedding dimension: {all_embeddings[0].shape[-1]}", flush=True)

        # Test similarity between embeddings
        print("\nTesting embedding similarities:", flush=True)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(all_embeddings[i]), torch.tensor(all_embeddings[j])
                ).item()
                print(f"Similarity between '{texts[i]}' and '{texts[j]}': {similarity:.4f}", flush=True)

        print("\nTest completed successfully!", flush=True)
        return True

    except Exception as e:
        print(f"\nError during test: {str(e)}", flush=True)
        import traceback

        print("\nFull traceback:", flush=True)
        print(traceback.format_exc(), flush=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
