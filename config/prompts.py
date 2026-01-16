"""
Configuration file for System Prompts and LLM Templates.
Centralizes the 'personality' of the DevSecOpsLM and the specific formatting
required by the underlying model architecture (e.g., Gemma, Llama).
"""

# --- 1. SYSTEM PERSONA ---
# This defines HOW the model should behave.
# It is prepended to the user instruction to condition the response.

SYSTEM_PROMPT_DEVSECOPS = (
    "You are DevSecOpsLM, an expert AI assistant specialized in DevOps, Security, "
    "and Cloud Infrastructure.\n"
    "Your core principles are:\n"
    "1. Security First: Always prioritize secure coding practices (OWASP Top 10).\n"
    "2. Automation: Prefer declarative solutions (Terraform, Kubernetes, Ansible) over manual steps.\n"
    "3. Observability: Ensure systems are monitorable and logged.\n"
    "4. No Secrets: NEVER generate or recommend hardcoded passwords or API keys.\n"
    "\n"
    "Answer the user's request precisely and technically."
)

# --- 2. TEACHER MODEL PROMPTS (Data Gen) ---
# Used in src/dataprep/02_generate_qa.py to generate synthetic data.

TEACHER_GENERATION_PROMPT = (
    "Analyze the following raw technical log or documentation:\n"
    "\"{raw_text}\"\n\n"
    "Task: Act as a Senior DevSecOps Engineer. Generate a realistic "
    "Instruction/Response pair based on this text.\n"
    "The 'Instruction' should be a question a junior engineer might ask.\n"
    "The 'Response' should be the technical answer derived strictly from the text.\n"
    "\n"
    "Output format (JSON only):\n"
    "{{\n"
    "  \"instruction\": \"...\",\n"
    "  \"response\": \"...\"\n"
    "}}"
)

# --- 3. MODEL TEMPLATES (Gemma Specific) ---
# Gemma uses specific control tokens: <start_of_turn>, <end_of_turn>.
# If you switch to Llama 3, you would only need to change this section.

class ModelTemplates:

    @staticmethod
    def gemma_chat(instruction: str, response: str = "", include_system: bool = True) -> str:
        """
        Formats the input into the official Gemma chat structure.

        Args:
            instruction: The user's query.
            response: The model's target response (empty during inference).
            include_system: Whether to inject the DevSecOps persona.
        """

        # 1. Prepare the User Turn
        user_content = instruction
        if include_system:
            # We inject the system prompt at the start of the first user message
            # because Gemma does not have a native <start_of_turn>system tag.
            user_content = f"{SYSTEM_PROMPT_DEVSECOPS}\n\nUser Query: {instruction}"

        # 2. Build the formatted string
        # <start_of_turn>user\n ... <end_of_turn>\n<start_of_turn>model\n ...
        formatted_text = (
            f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
            f"<start_of_turn>model\n{response}"
        )

        # 3. Add closing tag only if we are training (response is known)
        if response:
            formatted_text += "<end_of_turn>"

        return formatted_text

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Test the template output
    sample_instruction = "How do I rotate kube-apiserver certificates?"
    sample_response = "Use the kubeadm certs renew command."

    print("[:] Testing Gemma Template:")
    print(ModelTemplates.gemma_chat(sample_instruction, sample_response))
