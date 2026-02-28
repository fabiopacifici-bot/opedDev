Prototype app — opedDev

This folder contains a small Flask-based prototype to run inference against the fine-tuned model.

Files
- app.py — Flask application that exposes /generate and serves a simple UI at /
- static/index.html — the UI

How to run
1) Activate your conda environment (opedDev_py311):
   conda activate opedDev_py311

2) Install Flask if not already present:
   pip install flask

3) Start the app:
   python repositories/opedDev/prototype/app.py

4) Open your browser at http://localhost:7860

Notes
- The app lazily loads the tokenizer and model on first request. It will try to load the tokenizer from the HF hub (Qwen/Qwen3-0.6B) and the model weights from the local path (~/.openclaw/models/qwen3_webdev) if present.
- If the model is still being written by training, wait until training finishes or the checkpoint files are present before using the app.
- The UI is intentionally simple. For production use, wrap the model in a dedicated inference server (uvicorn/gunicorn), add batching, and secure the endpoint.
