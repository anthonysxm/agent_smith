import os

# --- BACKEND CONFIGURATION ---
# strictly define backend before importing Keras.
# JAX is recommended for high performance on GPUs/TPUs with XLA compilation.
os.environ["KERAS_BACKEND"] = "jax"
# Avoid preallocating 100% of GPU memory (JAX specific behavior)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import keras
import keras_nlp
import tensorflow as tf
import json

# --- HYPERPARAMETERS & PATHS ---
# Relative paths based on the file location to ensure portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "03_training", "final_instruct_dataset.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "production")
OUTPUT_FILENAME = "devsecops_adapter_v1.lora.h5"

# Model Configuration
PRESET = "gemma_2b_en"  # Lightweight model, suitable for tests and low-VRAM
LORA_RANK = 4           # Rank of the LoRA matrices (4, 8, 16)
SEQ_LENGTH = 512        # Context window size
BATCH_SIZE = 1          # Set to 1 for consumer GPUs, increase for A100/H100
EPOCHS = 1              # 1 epoch is usually sufficient for small datasets
LEARNING_RATE = 5e-5

# --- 1. UTILITY FUNCTIONS ---

def format_prompt(instruction, response):
    """
    Applies the specific chat template for the Gemma model.
    Structure: <start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>
    """
    template = (
        "<start_of_turn>user\n{instruction}<end_of_turn>\n"
        "<start_of_turn>model\n{response}<end_of_turn>"
    )
    return template.format(instruction=instruction, response=response)

def load_jsonl_dataset(filepath):
    """
    Python generator to read JSONL files line by line.
    Prevents RAM saturation by streaming data instead of loading it all at once.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Combine instruction and response using the prompt template
                full_text = format_prompt(data["instruction"], data["response"])
                yield full_text
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

# --- 2. MAIN EXECUTION ---

def main():
    print(f"[:] Starting DevSecOpsLM training pipeline")
    print(f"[:] Reading data from: {DATA_PATH}")

    # A. Mixed Precision Configuration (VRAM Efficiency)
    # Use "mixed_bfloat16" for Ampere+ GPUs (RTX 30xx/40xx, A100).
    # Fallback to "mixed_float16" if bfloat16 is not supported.
    keras.mixed_precision.set_global_policy("mixed_bfloat16")

    # B. Create tf.data Dataset (Pipeline)
    # from_generator enables streaming from the JSONL file
    raw_dataset = tf.data.Dataset.from_generator(
        lambda: load_jsonl_dataset(DATA_PATH),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    )

    # Batching and Prefetching
    # AUTOTUNE allows the CPU to prepare the next batch while GPU is training
    dataset = raw_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # C. Load Pre-trained Model
    print(f"[:] Loading model preset: {PRESET}...")
    llm = keras_nlp.models.GemmaCausalLM.from_preset(PRESET)

    # Set sequence length limit
    llm.preprocessor.sequence_length = SEQ_LENGTH

    # D. Enable LoRA (Low-Rank Adaptation)
    print(f"[:] Enabling LoRA (Rank={LORA_RANK})...")
    llm.backbone.enable_lora(rank=LORA_RANK)

    # Print summary to verify trainable parameters count
    llm.summary()

    # E. Compile Model
    # AdamW is the standard optimizer for Transformers
    # Excluding LayerNorm and Bias from weight decay is a best practice in NLP
    llm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        weighted_metrics=["accuracy"],
    )

    # F. Start Training
    print("[:] Starting training loop...")
    llm.fit(dataset, epochs=EPOCHS)

    # G. Export Artifacts
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    print(f"[:] Saving LoRA adapter weights to: {save_path}")

    # Save ONLY the LoRA weights (lightweight file)
    llm.backbone.save_lora_weights(save_path)

    print("[:] Process completed successfully.")

if __name__ == "__main__":
    main()
