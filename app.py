"""Leasely: AI lease analyzer with page-level citations and a risk scanner."""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from openai import OpenAI

# ---------- Config (set in .env or your shell) ----------
BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
API_KEY = os.getenv("NEBIUS_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

RISK_TOPICS = [
    "Automatic renewal and notice period to terminate",
    "Early termination penalties or break-lease fees",
    "Security deposit amount, deductions and refund timeline",
    "Rent increases and late fees",
    "Landlord entry and privacy rights",
    "Tenant liability for repairs and maintenance",
    "Subletting and guest restrictions",
    "Additional fees (maintenance, parking, utilities, admin)",
]

QA_PROMPT = """You are Leasely, an assistant that explains rental agreements.
Answer ONLY using the numbered excerpts provided. Never use outside knowledge about this lease.
If the excerpts do not contain the answer, set "found" to false and say the document does not cover it.
{style}
Return ONLY valid JSON, no markdown:
{{"found": true/false, "answer": "...", "citations": [{{"page": <int>, "quote": "<exact words copied from the excerpt, max 25 words>"}}]}}"""

RISK_PROMPT = """You review rental agreements for tenant risk, using ONLY the excerpts given.
Topic: {topic}
Return ONLY valid JSON, no markdown:
{{"found": true/false, "severity": "high"|"medium"|"low", "summary": "<one plain-English sentence>", "page": <int or null>, "quote": "<exact words from the excerpt, max 25 words>"}}
Severity: high = unusually harsh/costly/unfair to the tenant, medium = notable but common, low = standard/fair.
If the topic is not in the excerpts, set found to false."""


# ---------- Helpers ----------
def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_pages(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [{"page": i + 1, "text": p.get_text()} for i, p in enumerate(doc)]


def chunk_pages(pages, size=900, overlap=150):
    chunks = []
    for p in pages:
        text = re.sub(r"\s+", " ", p["text"]).strip()
        start = 0
        while start < len(text):
            chunks.append({"page": p["page"], "text": text[start : start + size]})
            start += size - overlap
    return chunks


def embed(texts, batch=32):
    vecs = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + batch])
        vecs.extend(d.embedding for d in resp.data)
    arr = np.array(vecs, dtype="float32")
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def build_index(pdf_bytes: bytes):
    chunks = chunk_pages(extract_pages(pdf_bytes))
    vecs = embed([c["text"] for c in chunks])
    return {"chunks": chunks, "vecs": vecs}


def retrieve(db, query, k=6):
    q = embed([query])
    scores = db["vecs"] @ q[0]
    ids = np.argsort(-scores)[:k]
    return [db["chunks"][i] for i in ids]


def format_excerpts(chunks):
    return "\n\n".join(f"[Excerpt {i+1} | Page {c['page']}]\n{c['text']}" for i, c in enumerate(chunks))


def chat_json(system, user):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return parse_json(resp.choices[0].message.content)


def quote_verified(quote, chunks):
    q = norm(quote or "")
    return bool(q) and any(q in norm(c["text"]) for c in chunks)


# ---------- Core features ----------
def answer_question(db, question, plain=True):
    chunks = retrieve(db, question)
    style = "Use simple, plain English a non-lawyer understands." if plain else "Keep the legal wording where useful."
    out = chat_json(QA_PROMPT.format(style=style), f"{format_excerpts(chunks)}\n\nQuestion: {question}")
    for c in out.get("citations", []):
        c["verified"] = quote_verified(c.get("quote"), chunks)
    return out


def scan_topic(db, topic):
    chunks = retrieve(db, topic, k=5)
    out = chat_json(RISK_PROMPT.format(topic=topic), format_excerpts(chunks))
    out["topic"] = topic
    out["verified"] = quote_verified(out.get("quote"), chunks)
    return out


def risk_scan(db):
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda t: scan_topic(db, t), RISK_TOPICS))
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(results, key=lambda r: order.get(r.get("severity", "low"), 3))


# ---------- UI ----------
st.set_page_config(page_title="Leasely", page_icon="🔍", layout="wide")
st.title("Leasely")
st.caption("Know what you're signing.")

if not API_KEY:
    st.error("Set NEBIUS_API_KEY first (see README).")
    st.stop()

with st.sidebar:
    st.header("Your lease")
    upload = st.file_uploader("Upload a lease (text-based PDF)", type="pdf")
    samples = sorted(Path("samples").glob("*.pdf")) if Path("samples").exists() else []
    sample = st.selectbox("...or use a sample", ["None"] + [s.name for s in samples])
    plain = st.toggle("Plain-English mode", value=True)

    pdf_bytes, name = None, None
    if upload:
        pdf_bytes, name = upload.getvalue(), upload.name
    elif sample != "None":
        pdf_bytes, name = (Path("samples") / sample).read_bytes(), sample

    if pdf_bytes and st.session_state.get("doc_name") != name:
        with st.spinner("Reading and indexing your lease..."):
            st.session_state.db = build_index(pdf_bytes)
            st.session_state.doc_name = name
            st.session_state.history = []
            st.session_state.risks = None
    if "db" in st.session_state:
        st.success(f"{st.session_state.doc_name}: {len(st.session_state.db['chunks'])} chunks indexed")

if "db" not in st.session_state:
    st.info("Upload a lease or pick a sample in the sidebar to begin.")
    st.stop()

db = st.session_state.db
tab_chat, tab_risk = st.tabs(["Ask a question", "Risk scan"])

SEV = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}

with tab_chat:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            for c in msg.get("citations", []):
                badge = "verified" if c.get("verified") else "unverified"
                st.markdown(f"> **Page {c.get('page')}** ({badge}): \"{c.get('quote')}\"")
    question = st.chat_input("e.g. Can my landlord keep my deposit? What if I leave early?")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.spinner("Searching your lease..."):
            out = answer_question(db, question, plain)
        answer = out.get("answer") or "Sorry, I couldn't process that. Try rephrasing."
        if out.get("found") is False:
            answer = "Not found in this document. " + answer
        st.session_state.history.append(
            {"role": "assistant", "content": answer, "citations": out.get("citations", []) if out.get("found") else []}
        )
        st.rerun()

with tab_risk:
    if st.button("Run risk scan", type="primary"):
        with st.spinner("Scanning key clauses..."):
            st.session_state.risks = risk_scan(db)
    for r in st.session_state.get("risks") or []:
        if not r.get("found"):
            st.markdown(f"**{r['topic']}**: not found in the document")
            continue
        label = SEV.get(r.get("severity", "low"), "Unknown")
        with st.expander(f"{label} | {r['topic']} (page {r.get('page')})", expanded=r.get("severity") == "high"):
            st.write(r.get("summary"))
            st.markdown(f"> \"{r.get('quote')}\" {'(verified)' if r.get('verified') else '(unverified)'}")