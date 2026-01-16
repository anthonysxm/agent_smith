import os
import contextlib

# --- BACKEND CONFIGURATION ---
# Set JAX as backend for inference (fastest on modern hardware)
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import keras
import keras_nlp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ADAPTER_PATH = os.path.join(BASE_DIR, "models", "production", "devsecops_adapter_v1.lora.h5")

# Global variable to store the model in memory
ml_models = {}

# --- UTILITIES ---
def format_prompt(instruction):
    """
    Formats the user input to match the training template (Gemma Chat).
    """
    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

# --- LIFESPAN MANAGER (Startup/Shutdown) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model only once when the API starts.
    This prevents reloading the heavy model for every request.
    """
    print("[:] API Startup: Loading Base Model (Gemma 2B)...")

    try:
        # 1. Load the base structure
        model = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

        # 2. Check if custom adapter exists
        if os.path.exists(ADAPTER_PATH):
            print(f"[:] Found DevSecOps Adapter at {ADAPTER_PATH}")
            print("[:] Injecting LoRA weights...")

            # Enable LoRA with the SAME rank used during training
            model.backbone.enable_lora(rank=4)

            # Load the learned weights
            model.backbone.load_lora_weights(ADAPTER_PATH)
            print("[:] Adapter loaded successfully.")
        else:
            print("[!] WARNING: Adapter file not found. Running with raw base model.")

        # 3. Optimize for inference (Compile)
        # compile() with 'jit_compile=True' optimizes XLA execution
        model.compile(sampler="top_k")

        # Store in global state
        ml_models["llm"] = model

        yield # Application runs here

    except Exception as e:
        print(f"[!] CRITICAL ERROR during model loading: {e}")
        raise e
    finally:
        # Cleanup (if needed)
        ml_models.clear()
        print("[:] API Shutdown: Resources released.")

# --- API DEFINITION ---
app = FastAPI(title="DevSecOpsLM Inference API", lifespan=lifespan)

class InferenceRequest(BaseModel):
    instruction: str
    max_length: int = 128

class InferenceResponse(BaseModel):
    response: str

# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Simple health check for Kubernetes/Docker."""
    if "llm" in ml_models:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """
    Main inference endpoint.
    Takes an instruction and returns the DevSecOps-tuned response.
    """
    model = ml_models.get("llm")
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Format input
    prompt = format_prompt(request.instruction)

    # 2. Run Inference
    # Note: KerasNLP generate() handles the tokenization internally
    try:
        output = model.generate(prompt, max_length=request.max_length)

        # 3. Clean output
        # Remove the prompt part to return only the generated answer
        # The model usually echoes the prompt, so we split it out.
        clean_response = output.replace(prompt, "").strip()

        return {"response": clean_response}

    except Exception as e:
        print(f"[!] Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run standalone for testing
    print("[:] Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
