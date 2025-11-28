from fastapi import FastAPI
from llama_cpp import Llama
import os

app = FastAPI()

model_path = os.getenv("MODEL_PATH", "models/mistral.gguf")

llm = Llama(model_path=model_path, n_ctx=4096)


@app.post("/ask")
async def ask(q: str):
    out = llm(q)
    return {"success": True, "response": out["choices"][0]['text']}