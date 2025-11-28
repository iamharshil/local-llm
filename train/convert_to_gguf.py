import os

# Clone llama.cpp (converter)
if not os.path.exists("llama.cpp"):
    os.system("git clone https://github.com/ggerganov/llama.cpp.git")

# Convert HF → GGUF
os.system("python3 llama.cpp/convert-hf-to-gguf.py models/lora-output --outfile models/mistral.gguf")
