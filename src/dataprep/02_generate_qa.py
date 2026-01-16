import os
import json
import sys
import time
from openai import OpenAI

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Import the centralized prompt template
from config.prompts import TEACHER_GENERATION_PROMPT

# --- CONFIGURATION ---
INPUT_FILE = os.path.join(project_root, "dataset", "02_sanitized", "chunks_sanitized.jsonl")
OUTPUT_FILE = os.path.join(project_root, "dataset", "03_training", "final_instruct_dataset.jsonl")

# Model Choice
TEACHER_MODEL = "gpt-4o-mini"

class QAGenerator:
    def __init__(self):
        # The client automatically reads OPENAI_API_KEY from env vars
        # Make sure to export it or use python-dotenv
        try:
            self.client = OpenAI()
            print(f"[:] QA Generator initialized with OpenAI model: {TEACHER_MODEL}")
        except Exception as e:
            print(f"[!] Error initializing OpenAI client: {e}")
            print("    Did you set the OPENAI_API_KEY environment variable?")
            sys.exit(1)

    def query_openai(self, chunk_text):
        """
        Sends the text chunk to GPT-4o-mini to generate a training pair.
        """
        full_prompt = TEACHER_GENERATION_PROMPT.format(raw_text=chunk_text)

        try:
            response = self.client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful data generation assistant. Output valid JSON only."},
                    {"role": "user", "content": full_prompt}
                ],
                response_format={"type": "json_object"}, # Critical: Forces valid JSON
                temperature=0.7 # Slight creativity for diverse questions
            )

            # Extract content
            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            print(f"    [!] OpenAI API Error: {e}")
            return None

    def generate_pair(self, chunk_text):
        return self.query_openai(chunk_text)

def main():
    print(f"[:] Starting ETL Step 2: Synthetic Data Generation (Powered by {TEACHER_MODEL})")

    if not os.path.exists(INPUT_FILE):
        print(f"[!] Error: Input file not found: {INPUT_FILE}")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    success_count = 0

    # Check for API Key existence before starting loop
    if not os.getenv("OPENAI_API_KEY"):
        print("[!] CRITICAL: OPENAI_API_KEY is missing from environment variables.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        for i, line in enumerate(f_in):
            try:
                record = json.loads(line)
                text_chunk = record.get("text", "")

                # Filter small noise
                if len(text_chunk) < 100:
                    continue

                print(f"    [:] Processing chunk #{i+1}...")

                qa_pair = generator.generate_pair(text_chunk)

                if qa_pair and "instruction" in qa_pair and "response" in qa_pair:
                    f_out.write(json.dumps(qa_pair) + "\n")
                    success_count += 1
                else:
                    print(f"    [?] Skipped invalid response for chunk #{i+1}")

            except json.JSONDecodeError:
                continue
            except KeyboardInterrupt:
                print("\n[!] Process interrupted by user.")
                break
            except Exception as e:
                print(f"    [!] Error on line {i}: {e}")

    print(f"[:] Step 2 Complete.")
    print(f"[:] Generated {success_count} training examples.")
    print(f"[:] Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generator = QAGenerator()
    main()
