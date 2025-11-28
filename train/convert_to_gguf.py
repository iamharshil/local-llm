import os

# Clone llama.cpp if missing
if not os.path.exists("llama.cpp"):
    os.system("git clone https://github.com/ggerganov/llama.cpp.git")

# Make sure converter is executable
os.system("chmod +x llama.cpp/convert-hf-to-gguf")

# Convert HuggingFace model → GGUF
cmd = "python3 llama.cpp/convert-hf-to-gguf --outfile models/mistral.gguf models/lora-output"
print("Running:", cmd)
os.system(cmd)
