# import os

# # Clone llama.cpp if missing
# if not os.path.exists("llama.cpp"):
#     os.system("git clone https://github.com/ggerganov/llama.cpp.git")

# # Make sure converter is executable
# os.system("chmod +x llama.cpp/convert-hf-to-gguf")

# # Convert HuggingFace model → GGUF
# cmd = "python3 llama.cpp/convert-hf-to-gguf --outfile models/mistral.gguf models/lora-output"
# print("Running:", cmd)
# os.system(cmd)


import os, sys, subprocess

# Clone llama.cpp if not exists
if not os.path.exists("llama.cpp"):
    subprocess.run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git"], check=True)

# Locate converter script
possible = [
    "llama.cpp/convert_hf_to_gguf.py",
    "llama.cpp/convert-hf-to-gguf.py",
    "llama.cpp/scripts/convert_hf_to_gguf.py",
    "llama.cpp/scripts/convert-hf-to-gguf.py",
]

converter = None
for p in possible:
    if os.path.exists(p):
        converter = p
        break

if converter is None:
    print("ERROR: could not find convert_hf_to_gguf script in llama.cpp")
    sys.exit(1)

hf_model_dir = "models/lora-output"
outfile = "models/phi3-mini.gguf"

cmd = [
    "python3", converter,
    hf_model_dir,
    "--outfile", outfile,
    "--outtype", "f16"   # or use q8_0 / q4_0 for quantized output
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("Conversion done. GGUF available at:", outfile)
