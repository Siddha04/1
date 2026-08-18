"""
Chat UI for the personal RAG assistant. Talks to the FastAPI backend over
HTTP — keeps the frontend completely stateless and swappable.
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Personal RAG Assistant", page_icon="🧠", layout="centered")
st.title("Personal RAG Assistant")
st.caption("Live web, market & sports data — grounded, cited answers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        st.success(f"Backend online · {health['knowledge_base_chunks']} chunks indexed")
    except Exception:
        st.error("Backend unreachable — is FastAPI running?")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s.get('source', 'knowledge base')}")

if query := st.chat_input("Ask about anything current — news, markets, scores..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Gathering live sources and thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat", json={"query": query}, timeout=120
                )
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.markdown(f"- {s.get('source', 'knowledge base')}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                    }
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
