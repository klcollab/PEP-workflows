import os

EMBEDDING_MODEL = "dunzhang/stella_en_1.5B_v"
GENAI_LLM = "Meta-Llama-3-70B-Instruct-Q5_K_S"
DEFAULT_DIR = "exit_interviews/"
LLM_TEMPERATURE = 0.0  # For (hopeful) deterministic output

# Ollama Model name
model = os.environ["GENAI_LLM"] if "GENAI_LLM" in os.environ else GENAI_LLM

# Path to directory containing txt files of interview transcripts
exit_interview_dir = os.environ["INTERVIEW_DIR"] if "INTERVIEW_DIR" in os.environ else DEFAULT_DIR
