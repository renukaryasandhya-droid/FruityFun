from __future__ import annotations

import streamlit as st

from fruity_fun.agent import FruityFunAgent, citation_rows
from fruity_fun.config import get_settings
from fruity_fun.corpus import load_chunks
from fruity_fun.retrieval import HybridRetriever

st.set_page_config(page_title="Fruity Fun", page_icon="🍓", layout="centered")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(150deg, #fff8dc 0%, #fff 48%, #e8f8e8 100%)}
    .hero {padding: 1.2rem 1.4rem; border-radius: 24px; background: #ff6b6b; color: white;
           box-shadow: 0 12px 30px #ff6b6b33; margin-bottom: 1rem}
    .hero h1 {margin: 0; font-size: 2.5rem}.hero p {font-size: 1.08rem; margin-bottom: 0}
    [data-testid="stChatMessage"] {border-radius: 20px; background: #ffffffcc}
    </style>
    <div class="hero"><h1>🍓 Fruity Fun</h1>
    <p>Ask about one fruit—or invent a whole fruit-party picture!</p></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def make_agent() -> FruityFunAgent:
    settings = get_settings()
    chunks = load_chunks(settings.processed_corpus_path)
    return FruityFunAgent(settings, HybridRetriever(settings, chunks))


settings = get_settings()
agent = make_agent()

with st.sidebar:
    st.header("Fruit library")
    chunk_count = len(agent.retriever.chunks)
    st.metric("Ready-to-search passages", chunk_count)
    if settings.missing_runtime_secrets:
        st.warning("Setup needed: " + ", ".join(settings.missing_runtime_secrets))
    if not chunk_count:
        st.info("Add PDFs to data/pdfs, then run the ingestion command from the README.")
    st.caption("Answers use your PDF library. Pictures use a safe imagination path.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image"):
            st.image(message["image"], caption="A Fruity Fun picture")

examples = (
    "Try: ‘Why do strawberries have seeds outside?’ or "
    "‘Draw mango, kiwi, and grapes having a picnic.’"
)
prompt = st.chat_input("Which fruit shall we explore?")
if not st.session_state.messages:
    st.caption(examples)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Picking the juiciest facts and painting a picture…"):
            result = agent.invoke(prompt)
        st.markdown(result["answer"])
        if result.get("image_path"):
            st.image(result["image_path"], caption="A Fruity Fun picture")
        hits = result.get("reranked_hits", [])
        if hits:
            with st.expander("Where did these facts come from?"):
                st.dataframe(citation_rows(hits), use_container_width=True, hide_index=True)
                st.caption(f"Retrieval confidence: {result.get('confidence', 0):.0%}")
        if result.get("warnings"):
            with st.expander("Technical notes"):
                for warning in result["warnings"]:
                    st.caption(warning)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "image": result.get("image_path"),
        }
    )
