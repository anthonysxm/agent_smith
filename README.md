# Agent Smith LLM

Specialized, secure-by-design Low-Rank Adaptation (LoRA) fine-tuning pipeline for Large Language Models.

It leverages **Keras 3** and **JAX** to fine-tune the **Gemma 2 (2B Instruct)** model on private technical documentation and logs. The pipeline includes a dedicated sanitization layer to prevent PII (Personally Identifiable Information) and secret leaks during the training process.



## 🚀 Key Features

* **Security First:** Integrated `DataSanitizer` removes IPs, emails, MAC addresses, and potential API keys before the model sees the data.
* **Efficient Training:** Uses LoRA (Low-Rank Adaptation) and Mixed Precision (bfloat16) to train on consumer-grade GPUs.
* **High Performance:** Powered by the JAX backend with XLA compilation for maximum throughput.
* **Production Ready:** Includes a FastAPI inference server implementing the Adapter Pattern for hot-loading weights.
* **Scalable Data Pipeline:** Streams data using `tf.data` to handle datasets larger than RAM.

## 📂 Project Structure

```
devsecops-lm-project/
├── data/                          # Data storage (GitIgnored)
│   ├── 01_raw/                    # Place your raw logs/docs here
│   ├── 02_sanitized/              # Intermediate cleaned chunks
│   └── 03_training/               # Final Instruction/Response dataset
├── models/                        # Artifact storage
│   └── production/                # Generated LoRA adapters (.h5)
├── src/
│   ├── security/                  # PII Sanitization logic
│   ├── dataprep/                  # ETL Scripts (Chunking & QA Generation)
│   ├── training/                  # Keras/JAX Fine-tuning script
│   └── serving/                   # FastAPI Inference Server
├── config/                        # System prompts and templates
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

🛠️ Prerequisites

Python: 3.10 or higher.

GPU: NVIDIA GPU with at least 6GB VRAM (for Gemma 2B).

Drivers: CUDA 12.x installed.

📦 Installation

Clone the repository:

Create a Virtual Environment:
```
python -m venv venv
source venv/bin/activate
```

Install Dependencies:
```
pip install -r requirements.txt
pip install -U "jax[cuda12]"
```

⚡ Workflow & Usage
1. Data Preparation (ETL)

Place your raw .txt or .log files in data/01_raw/.

Step A: Sanitize and Chunk Cleans PII and splits large files into manageable text blocks.

```
python src/dataprep/01_chunk_and_clean.py
```

Step B: Generate Synthetic Instructions Transforms raw text chunks into Q&A pairs (Instruction/Response) for the Instruct model.

```
python src/dataprep/02_generate_qa.py
```

2. Fine-Tuning (Training)

Starts the JAX-based training pipeline using the Gemma 2 2B Instruct preset. Artifacts are saved to models/production/.

```
python src/training/fine_tune_keras.py
```

3. Inference (Serving)

Launches the FastAPI server. It loads the base model and injects your custom LoRA adapter on startup.

```
python src/serving/inference_api.py
```

Test the API:

```
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"instruction": "How do I secure a Docker container?", "max_length": 128}'
```

⚙️ Configuration

Prompts: Modify config/prompts.py to change the System Prompt ("Persona") of the model.

Hyperparameters: Adjust src/training/fine_tune_keras.py (Batch size, LoRA Rank, Learning Rate) to fit your hardware capabilities.

🛡️ Security Disclaimer

While this pipeline includes a DataSanitizer based on Regex patterns, it is a defense-in-depth measure.

Do not rely solely on it. Always audit your dataset for sensitive secrets before ingestion.

Generated models may still hallucinate or produce insecure code suggestions. Always review AI-generated output.

📄 License

MIT License
