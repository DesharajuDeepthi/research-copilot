import time

import httpx
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Research Copilot", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []


def trace_icon(line: str) -> str:
    lower = line.lower()
    if lower.startswith("retrieve"):
        return "🔍"
    if lower.startswith("graded"):
        return "📊"
    if lower.startswith("query rewritten"):
        return "✏️"
    if lower.startswith("live api fallback"):
        return "🌐"
    if lower.startswith("synthesize"):
        return "🧠"
    if lower.startswith("check_grounding") or lower.startswith("validate_citations"):
        return "❌" if "fail" in lower else "✅"
    return "•"


def ask_backend(query: str) -> dict:
    response = httpx.post(API_URL, json={"query": query}, timeout=120)
    response.raise_for_status()
    return response.json()


def render_badges(message: dict) -> None:
    def badge(text: str, color: str) -> str:
        return (
            f'<span style="background-color:{color};color:white;padding:2px 8px;'
            f'border-radius:8px;margin-right:6px;font-size:0.8em;">{text}</span>'
        )

    badges = [
        badge("Grounding ✓", "#16a34a") if message.get("grounding_passed") else badge("Grounding ✗", "#dc2626"),
        badge("Citations ✓", "#16a34a") if message.get("citation_validated") else badge("Citations ✗", "#dc2626"),
    ]
    if message.get("fallback_used"):
        badges.append(badge("Fallback Used", "#ea580c"))

    st.markdown(" ".join(badges), unsafe_allow_html=True)


def render_sources(citations: list[dict]) -> None:
    if not citations:
        st.caption("No sources cited.")
        return

    df = pd.DataFrame(
        [
            {
                "Title": citation.get("title"),
                "Year": citation.get("year"),
                "Cited By": citation.get("cited_by"),
                "DOI": citation.get("doi"),
            }
            for citation in citations
        ]
    )
    st.dataframe(
        df,
        hide_index=True,
        column_config={"DOI": st.column_config.LinkColumn("DOI")},
    )


def render_trace(placeholder, lines: list[str]) -> None:
    placeholder.markdown("\n\n".join(f"{trace_icon(line)} {line}" for line in lines))


chat_col, trace_col = st.columns([0.6, 0.4])

with chat_col:
    st.title("Research Copilot — Data Engineering & Analytics")
    st.caption("Powered by OpenAlex + Corrective RAG")

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_badges(message)
                with st.expander("Sources"):
                    render_sources(message.get("citations", []))

    query = st.chat_input("Ask a research question...")

with trace_col:
    st.title("Agent Reasoning Trace")
    trace_placeholder = st.empty()

    if not query:
        last_trace = next(
            (m["agent_trace"] for m in reversed(st.session_state.history) if m["role"] == "assistant"),
            [],
        )
        if last_trace:
            render_trace(trace_placeholder, last_trace)

if query:
    st.session_state.history.append({"role": "user", "content": query})

    with chat_col:
        with st.chat_message("user"):
            st.markdown(query)

    with chat_col:
        with st.spinner("Thinking..."):
            result = ask_backend(query)

    revealed = []
    for line in result["agent_trace"]:
        revealed.append(line)
        render_trace(trace_placeholder, revealed)
        time.sleep(0.4)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "agent_trace": result["agent_trace"],
            "fallback_used": result["fallback_used"],
            "grounding_passed": result["grounding_passed"],
            "citation_validated": result["citation_validated"],
        }
    )

    st.rerun()
