from __future__ import annotations

import os
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Chat", page_icon="💬", layout="centered")
st.title("Chat")
st.caption("Frontend en Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu mensaje"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "message": prompt,
        "session_id": st.session_state.session_id,
        "history": st.session_state.messages[:-1],
    }

    try:
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = httpx.post(f"{API_URL}/chat", json=payload, timeout=60.0)
                response.raise_for_status()
                data = response.json()
                st.session_state.session_id = data["session_id"]
                answer = data["reply"]
                st.markdown(answer)
                if data.get("sources"):
                    st.caption(
                        "Fuentes: "
                        + " | ".join(
                            f"{item['title']} - {item['url']}" for item in data["sources"][:3]
                        )
                    )
                if data.get("escalated"):
                    st.info("Caso marcado para revision humana.")
    except httpx.HTTPError as exc:
        answer = f"Error conectando con el backend: {exc}"
        with st.chat_message("assistant"):
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
