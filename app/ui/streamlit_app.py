from __future__ import annotations

import os
import time
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# 1. La configuración de la página DEBE ser el primer comando de Streamlit
st.set_page_config(page_title="Chat Agencia de Viajes", page_icon="✈️", layout="centered")

# 2. Leer el ID directamente de los parámetros de la URL (si existe)
url_session = st.query_params.get("session_id", "")
print(f"ID de sesión desde URL: {url_session}")

with st.sidebar:
    st.header("⚙️ Recuperación de Sesión")
    st.warning("Streamlit reinicia la sesión al cerrar la pestaña. Guarda el ID a continuación para recuperar tu historial luego.")
    
    # 3. Campo de texto para que el usuario pueda ver su ID actual o pegar uno anterior
    user_input_id = st.text_input(
        "Tu ID de Sesión (Memori):", 
        value=st.session_state.get("session_id", url_session)
    )

# 4. Lógica de asignación del ID
if user_input_id:
    # Si hay un ID en el cuadro de texto (escrito o cargado desde la URL), lo usamos
    st.session_state.session_id = user_input_id
elif "session_id" not in st.session_state:
    # Si todo está vacío (primera vez que entra), generamos uno nuevo aleatorio
    st.session_state.session_id = str(uuid4())

# 5. ¡TRUCO CLAVE! Inyectar el ID activo en la URL del navegador.
# Así, si recargas la página (F5), leerá el ID desde la URL.
st.query_params["session_id"] = st.session_state.session_id

# 6. Limpiar el historial visual de la pantalla si el usuario cambia de ID
if "last_session_id" not in st.session_state or st.session_state.last_session_id != st.session_state.session_id:
    st.session_state.messages = []
    st.session_state.last_session_id = st.session_state.session_id


st.title("✈️ Chat")
st.caption("Agente conectado a RAG y Memori")

# --- LÓGICA DEL CHAT ---
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
                start_time = time.time() # 1. Inicia el cronómetro
                response = httpx.post(f"{API_URL}/chat", json=payload, timeout=60.0)
                response.raise_for_status()
                end_time = time.time() # 2. Detiene el cronómetro
                tiempo_total = end_time - start_time # 3. Calcula la diferencia
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
                
                # Expandible para trazabilidad y evaluación
                with st.expander("🔍 Evaluación del Agente y Memoria"):
                    st.write(f"**ID de Sesión (Memori):** `{data['session_id']}`")
                    st.write(f"**Tiempo de respuesta:** `{tiempo_total:.2f} segundos`")
                    st.write(f"**Estado:** {'Escalado a humanos' if data.get('escalated') else 'Resuelto por IA'}")
                    
                    if data.get("sources"):
                        st.markdown("**Fuentes utilizadas (RAG/Web):**")
                        for item in data["sources"]:
                            st.caption(f"- {item['title']} ({item['url']})")
                    else:
                        st.caption("No se requirieron documentos RAG adicionales para esta respuesta.")
                        
                if data.get("escalated"):
                    st.info("Caso marcado para revision humana.")
    except httpx.HTTPError as exc:
        answer = f"Error conectando con el backend: {exc}"
        with st.chat_message("assistant"):
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
