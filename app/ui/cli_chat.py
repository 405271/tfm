from __future__ import annotations

import os
from uuid import uuid4

import httpx
from dotenv import load_dotenv

try:
    import readline
except ImportError:  # pragma: no cover - unavailable on some platforms
    readline = None


load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def _configure_input_history() -> None:
    if readline is None:
        return

    readline.parse_and_bind("tab: complete")


def main() -> None:
    _configure_input_history()

    history: list[dict[str, str]] = []
    session_id = str(uuid4())

    print("Chat CLI")
    print("Escribe tu mensaje. Usa /exit para salir y /clear para limpiar el historial.")

    while True:
        try:
            message = input("\nTu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break

        if not message:
            continue
        if message == "/exit":
            print("Saliendo.")
            break
        if message == "/clear":
            history.clear()
            session_id = str(uuid4())
            if readline is not None:
                readline.clear_history()
            print("Historial limpiado.")
            continue

        if readline is not None:
            readline.add_history(message)

        payload = {
            "message": message,
            "session_id": session_id,
            "history": history,
        }

        try:
            response = httpx.post(f"{API_URL}/chat", json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            session_id = data["session_id"]
            answer = data["reply"]
        except httpx.HTTPError as exc:
            print(f"Error conectando con el backend: {exc}")
            continue

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        print(f"Bot: {answer}")
        if data.get("sources"):
            print("Fuentes:")
            for item in data["sources"][:3]:
                print(f"- {item['title']}: {item['url']}")
        if data.get("escalated"):
            print("Sistema: caso marcado para revision humana.")


if __name__ == "__main__":
    main()
