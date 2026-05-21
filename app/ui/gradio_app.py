from __future__ import annotations

import os
from uuid import uuid4

import gradio as gr
import httpx
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return str(content)


def _normalize_history(history: list[dict[str, object]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for item in history or []:
        role = item.get("role")
        content = _content_to_text(item.get("content"))
        if role in {"user", "assistant", "system"} and content:
            normalized.append({"role": role, "content": content})

    return normalized


def chat_with_api(
    message: str,
    history: list[dict[str, object]] | None,
    session_id: str | None,
) -> tuple[str, list[dict[str, str]], str]:
    if not message.strip():
        return "", _normalize_history(history), session_id or str(uuid4())

    normalized_history = _normalize_history(history)
    session_id = session_id or str(uuid4())

    payload = {
        "message": message,
        "session_id": session_id,
        "history": normalized_history,
    }

    try:
        response = httpx.post(f"{API_URL}/chat", json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        answer = data["reply"]
        session_id = data["session_id"]
        if data.get("sources"):
            answer = (
                f"{answer}\n\nFuentes:\n"
                + "\n".join(f"- {item['title']}: {item['url']}" for item in data["sources"][:3])
            )
        if data.get("escalated"):
            answer = f"{answer}\n\n[Caso marcado para revision humana]"
    except httpx.HTTPError as exc:
        answer = f"Error conectando con el backend: {exc}"

    updated_history = normalized_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", updated_history, session_id


with gr.Blocks(title="Chat") as demo:
    gr.Markdown("# Chat")
    gr.Markdown("Frontend en Gradio")

    chatbot = gr.Chatbot(height=500, value=[])
    session_state = gr.State(str(uuid4()))
    message = gr.Textbox(
        label="Mensaje",
        placeholder="Escribe tu mensaje",
        lines=3,
    )

    with gr.Row():
        send_button = gr.Button("Enviar", variant="primary")
        clear_button = gr.Button("Limpiar")

    send_button.click(
        fn=chat_with_api,
        inputs=[message, chatbot, session_state],
        outputs=[message, chatbot, session_state],
    )
    message.submit(
        fn=chat_with_api,
        inputs=[message, chatbot, session_state],
        outputs=[message, chatbot, session_state],
    )
    clear_button.click(lambda: ("", [], str(uuid4())), outputs=[message, chatbot, session_state])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
