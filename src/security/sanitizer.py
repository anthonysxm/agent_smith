import re

class DataSanitizer:
    """
    Handles the removal of sensitive information from raw text data
    before it enters the machine learning pipeline.
    """

    def __init__(self):
        # Define Regex patterns for sensitive data detection
        self.patterns = {
            # 1. IPv4 Addresses (e.g., 192.168.1.1)
            # Matches standard dot-decimal notation
            "IPV4": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',

            # 2. Email Addresses (e.g., admin@corp.local)
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',

            # 3. Generic Secrets/Tokens (Heuristic)
            # Detects patterns like "api_key=...", "token: ...", "secret_id : ..."
            # followed by a long alphanumeric string (20+ chars)
            "SECRET_KEY": r'(?i)\b(api[_-]?key|access[_-]?token|secret|password|auth)\s*[:=]\s*[A-Za-z0-9_\-]{16,}\b',

            # 4. MAC Addresses (e.g., 00:1A:2B:3C:4D:5E)
            "MAC_ADDR": r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})'
        }

        # Define replacements
        self.replacements = {
            "IPV4": "[REDACTED_IP]",
            "EMAIL": "[REDACTED_EMAIL]",
            "SECRET_KEY": "[REDACTED_SECRET]",
            "MAC_ADDR": "[REDACTED_MAC]"
        }

    def clean_text(self, text: str) -> str:
        """
        Applies all regex patterns to the input text and replaces matches
        with their corresponding placeholders.
        """
        if not text:
            return ""

        cleaned_text = text

        # Iterate through all defined patterns
        for key, pattern in self.patterns.items():
            replacement = self.replacements.get(key, "[REDACTED]")
            cleaned_text = re.sub(pattern, replacement, cleaned_text)

        return cleaned_text

# --- UNIT TEST (Standalone execution) ---
if __name__ == "__main__":
    # Test data with fake secrets
    raw_log = (
        "Error at 2023-10-12 08:00:00. "
        "Host 192.168.0.55 failed to connect. "
        "User admin@company.com attempted login with api_key=sk-abC12345678901234567890abcdef. "
        "MAC: 00:1B:44:11:3A:B7."
    )

    print("[:] Testing Sanitizer...")
    print(f"[:] Original: \n{raw_log}\n")

    sanitizer = DataSanitizer()
    clean_log = sanitizer.clean_text(raw_log)

    print(f"[:] Sanitized: \n{clean_log}")

    # Validation assertion
    assert "192.168.0.55" not in clean_log
    assert "admin@company.com" not in clean_log
    assert "sk-abC" not in clean_log
    print("\n[V] Security Check Passed: No leaks detected.")
