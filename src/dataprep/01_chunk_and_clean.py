import os
import json
import sys

# --- PATH SETUP ---
# Add the project root to sys.path to allow importing from src.security
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.security.sanitizer import DataSanitizer

# --- CONFIGURATION ---
# Input: Raw messy logs/docs
INPUT_DIR = os.path.join(project_root, "dataset", "01_raw")
# Output: Cleaned, chunked text (intermediate state)
OUTPUT_FILE = os.path.join(project_root, "dataset", "02_sanitized", "chunks_sanitized.jsonl")

# Chunking settings
CHUNK_SIZE = 500  # Number of words per block
OVERLAP = 50      # Overlap to maintain context between blocks

def create_chunks(text, chunk_size, overlap):
    """
    Splits a long text into overlapping windows.
    This ensures that context isn't lost if a sentence is cut in the middle.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    # Sliding window logic
    for i in range(0, len(words), chunk_size - overlap):
        # Join words back into a string
        chunk = " ".join(words[i : i + chunk_size])

        # Filter out very short chunks (e.g., end of file artifacts)
        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks

def main():
    print(f"[:] Starting ETL Step 1: Clean & Chunk")
    print(f"[:] Input Directory: {INPUT_DIR}")

    # 1. Initialize Security Layer
    sanitizer = DataSanitizer()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total_files = 0
    total_chunks = 0

    # 2. Open Output Stream
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        # 3. Walk through the raw data folder
        for root, _, files in os.walk(INPUT_DIR):
            for filename in files:
                # Filter for text-based files only
                if not filename.lower().endswith(('.txt', '.log', '.md', '.json')):
                    continue

                filepath = os.path.join(root, filename)

                try:
                    # A. Read Raw Content
                    # errors='ignore' prevents crashing on binary garbage in logs
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f_in:
                        raw_content = f_in.read()

                    if not raw_content.strip():
                        continue

                    # B. Sanitize (Security Check)
                    # Remove IPs, Emails, Secrets BEFORE chunking
                    clean_content = sanitizer.clean_text(raw_content)

                    # C. Create Chunks
                    chunks = create_chunks(clean_content, CHUNK_SIZE, OVERLAP)

                    # D. Write to JSONL
                    for chunk in chunks:
                        record = {
                            "source": filename, # Good for traceability
                            "text": chunk
                        }
                        f_out.write(json.dumps(record) + "\n")
                        total_chunks += 1

                    total_files += 1
                    print(f"    [+] Processed: {filename} ({len(chunks)} chunks)")

                except Exception as e:
                    print(f"    [!] Error processing {filename}: {e}")

    print(f"[:] Step 1 Complete.")
    print(f"[:] Processed {total_files} files into {total_chunks} chunks.")
    print(f"[:] Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
