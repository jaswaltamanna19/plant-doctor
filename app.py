from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# Load env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health() -> tuple[Any, int]:
    return jsonify({
        "status": "ok",
        "model": OPENROUTER_MODEL
    }), 200


@app.post("/api/chat")
def chat() -> tuple[Any, int]:
    if not OPENROUTER_API_KEY:
        return jsonify({
            "error": "OPENROUTER_API_KEY missing in .env"
        }), 500

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    image_base64 = data.get("image_base64")
    image_mime = data.get("image_mime", "image/jpeg")

    if not message and not image_base64:
        return jsonify({"error": "Send text or image"}), 400

    # System prompt
    system_prompt = (
        "You are a plant disease detection and care assistant. "
        "Analyze plant symptoms and provide:\n"
        "1. Disease/issue\n"
        "2. Confidence\n"
        "3. Causes\n"
        "4. Treatment steps\n"
        "5. Prevention tips\n"
        "Keep response clear and practical."
    )

    # Build message content
    user_content = []

    if message:
        user_content.append({
            "type": "text",
            "text": message
        })
    else:
        user_content.append({
            "type": "text",
            "text": "Analyze this plant image and give diagnosis."
        })

    if image_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_mime};base64,{image_base64}"
            }
        })

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.4,
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",  # your frontend port
        "X-Title": "Plant AI Assistant"
    }

    try:
        res = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        res.raise_for_status()
        result = res.json()

        reply = result["choices"][0]["message"]["content"]

        return jsonify({"reply": reply}), 200

    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "OpenRouter request failed",
            "details": str(e)
        }), 502

    except Exception as e:
        return jsonify({
            "error": "Server error",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)