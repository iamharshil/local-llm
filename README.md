# Local-LLM

Train and serve a locally-run LLM using LoRA fine-tuning and a FastAPI inference API.

**Repository layout**
- **`api/`**: FastAPI server code (`server.py`).
- **`train/`**: Training utilities and conversion scripts.
- **`models/`**: Trained and converted model artifacts.
- **`data/`**: Training dataset (e.g. `dataset.jsonl`).

**Quick Start (macOS / Linux)**

Prerequisites:
- Docker and Docker Compose installed and working.
- Sufficient disk space and RAM for model training/inference.

1) Build the Docker image used for training and inference:

```bash
docker compose build
```

2) Train / fine-tune the model (runs the training container):

```bash
docker compose run --rm llm-train
```

The fine-tuned model will be saved under `models/lora-output/` (path may vary depending on training configuration).

3) Convert the fine-tuned model to GGUF for efficient inference (optional but recommended):

```bash
docker compose run --rm llm-train python train/convert_to_gguf.py
```

The conversion script produces a file such as `models/mistral.gguf` (name depends on your base model and script output).

4) Start the FastAPI inference server:

```bash
docker compose up -d llm-api
```

By default the API listens on `http://localhost:8000` (confirm in `docker-compose.yml`).

**API Usage Examples**

Simple `curl` request (JSON):

```bash
curl -sS -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"q": "How many shirts from 30 meters?"}'
```

Example Python client using `requests`:

```python
import requests

resp = requests.post('http://localhost:8000/ask', json={'q': 'How many shirts from 30 meters?'})
print(resp.json())
```

Adjust the endpoint and payload according to the API implemented in `api/server.py`.

**Configuration & Notes**
- Check `docker-compose.yml` to confirm service names, port mappings, and volume mounts.
- Use `docker compose run --rm llm-train` to run one-off training tasks without keeping the container.
- The `llm-api` service will normally mount `models/` so converted GGUF or LoRA outputs are available to the server.

**Troubleshooting**
- If Docker Compose errors, ensure you are using a compatible Docker Engine and the Compose plugin: `docker compose version`.
- If the model file is missing, confirm the training step completed and outputs were saved to `models/`.
- Logs:

```bash
docker compose logs -f llm-api
docker compose logs -f llm-train
```

**Development tips**
- To iterate on `api/server.py` locally without Docker, create a virtualenv and install requirements (if present) and run `uvicorn api.server:app --reload`.
- When training large models, prefer running on a machine with a capable GPU and enough memory.

**Files to check**
- `api/server.py`: API routes and request/response format.
- `train/convert_to_gguf.py`: Conversion script (arguments or output names).
- `train/train.py`: Training loop and save locations.

If you want, I can also:
- Add a short example `curl` test into a `tests/` script.
- Update `docker-compose.yml` with explicit environment variables and comments.


**Troubleshooting**
- If Docker Compose errors, ensure you are using a compatible Docker Engine and the Compose plugin: `docker compose version`.
- If the model file is missing, confirm the training step completed and outputs were saved to `models/`.

- Check running containers:
```bash
docker compose ps
```
 
- Logs:
```bash
  docker compose logs -f llm-api
  docker compose logs -f llm-train
```

---
Updated: concise setup, commands, API examples, and troubleshooting.
