#!/usr/bin/env python3
"""
Prototype inference app for the fine-tuned Qwen model.

Usage:
  # in your conda env (opedDev_py311)
  python repositories/opedDev/prototype/app.py

This starts a small Flask server at http://0.0.0.0:7860 with a simple UI at /
The server lazily loads the tokenizer and model on first request. It tries to load the tokenizer from the HF hub
(Qwen/Qwen3-0.6B) and model weights from MODEL_PATH (default: ~/.openclaw/models/qwen3_webdev).
If MODEL_PATH is missing, it will fall back to loading the base model from HF (may download large weights).

This file is intentionally minimal and robust: it will not crash if the model isn't available yet; instead it
returns a helpful error message and keeps the server running so you can retry once training finishes.
"""
import os
import threading
import logging
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="/static")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prototype")

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/.openclaw/models/qwen3_webdev"))
HF_BASE = os.environ.get("HF_BASE_MODEL", "Qwen/Qwen3-0.6B")

# Lazy-loaded globals
tokenizer = None
model = None
_model_lock = threading.Lock()

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    logger.warning("Missing transformers/torch in the active Python env: %s", e)
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


def load_model_and_tokenizer():
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return True, "already_loaded"
    with _model_lock:
        if tokenizer is not None and model is not None:
            return True, "already_loaded"
        try:
            # Load tokenizer: prefer local model tokenizer if present, otherwise HF hub
            if os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
                    logger.info("Loaded tokenizer from local model path: %s", MODEL_PATH)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(HF_BASE, trust_remote_code=True)
                    logger.info("Loaded tokenizer from HF base: %s", HF_BASE)
            else:
                tokenizer = AutoTokenizer.from_pretrained(HF_BASE, trust_remote_code=True)
                logger.info("Loaded tokenizer from HF base: %s", HF_BASE)

            # Load model: prefer local weights if present
            device_dtype = None
            if torch is not None and torch.cuda.is_available():
                device_dtype = getattr(torch, "float16")
            else:
                device_dtype = getattr(torch, "float32") if torch is not None else None

            if os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH):
                model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map='auto', torch_dtype=device_dtype)
                logger.info("Loaded model from local path: %s", MODEL_PATH)
            else:
                model = AutoModelForCausalLM.from_pretrained(HF_BASE, trust_remote_code=True, device_map='auto', torch_dtype=device_dtype)
                logger.info("Loaded model from HF base: %s", HF_BASE)

            return True, "loaded"
        except Exception as e:
            logger.exception("Failed to load model/tokenizer: %s", e)
            tokenizer = None
            model = None
            return False, str(e)


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/health')
def health():
    ok = (tokenizer is not None and model is not None)
    info = {
        'model_path': MODEL_PATH,
        'tokenizer_loaded': tokenizer is not None,
        'model_loaded': model is not None,
    }
    return jsonify({'ok': ok, 'info': info})


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json or {}
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'missing prompt'}), 400
    max_new_tokens = int(data.get('max_new_tokens', 64))
    temperature = float(data.get('temperature', 0.0))

    # Ensure model loaded
    loaded, msg = load_model_and_tokenizer()
    if not loaded:
        return jsonify({'error': 'model not loaded', 'detail': msg}), 500

    try:
        device = next(model.parameters()).device
    except Exception:
        device = None

    try:
        inputs = tokenizer(prompt, return_tensors='pt')
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=(temperature>0.0), temperature=temperature)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return jsonify({'generated': text})
    except Exception as e:
        logger.exception('Generation failed')
        return jsonify({'error': 'generation_failed', 'detail': str(e)}), 500


if __name__ == '__main__':
    host = os.environ.get('PROTOTYPE_HOST', '0.0.0.0')
    port = int(os.environ.get('PROTOTYPE_PORT', 7860))
    logger.info('Starting prototype app at http://%s:%d – MODEL_PATH=%s', host, port, MODEL_PATH)
    app.run(host=host, port=port)
