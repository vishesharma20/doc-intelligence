import streamlit as st
import tempfile
import os

from app.loaders import load_document, chunk_text
from app.orchestrator import run_pipeline
from app.agents.qa import ask

st.set_page_config(page_title="Document Intelligence System", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
    margin: 0 auto;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 20px 0 10px;">
<svg width="110" viewBox="0 0 680 320" xmlns="http://www.w3.org/2000/svg">
<line x1="340" y1="40" x2="340" y2="65" stroke="#7F77DD" stroke-width="4" stroke-linecap="round"/>
<circle cx="340" cy="36" r="9" fill="#7F77DD"/>
<rect x="255" y="65" width="170" height="150" rx="45" fill="#EEEDFE" stroke="#534AB7" stroke-width="2"/>
<circle cx="305" cy="135" r="20" fill="#26215C"/>
<circle cx="313" cy="128" r="6" fill="#E1F5EE"/>
<circle cx="375" cy="135" r="20" fill="#26215C"/>
<circle cx="383" cy="128" r="6" fill="#E1F5EE"/>
<path d="M320 175 Q340 190 360 175" fill="none" stroke="#26215C" stroke-width="4" stroke-linecap="round"/>
<rect x="285" y="230" width="110" height="80" rx="20" fill="#9FE1CB" stroke="#0F6E56" stroke-width="2"/>
<rect x="315" y="250" width="50" height="40" rx="4" fill="#04342C"/>
<line x1="320" y1="260" x2="360" y2="260" stroke="#9FE1CB" stroke-width="2"/>
<line x1="320" y1="270" x2="360" y2="270" stroke="#9FE1CB" stroke-width="2"/>
<line x1="320" y1="280" x2="345" y2="280" stroke="#9FE1CB" stroke-width="2"/>
<circle cx="240" cy="270" r="10" fill="#D4537E"/>
<circle cx="440" cy="270" r="10" fill="#D4537E"/>
<line x1="255" y1="240" x2="240" y2="270" stroke="#D4537E" stroke-width="6" stroke-linecap="round"/>
<line x1="425" y1="240" x2="440" y2="270" stroke="#D4537E" stroke-width="6" stroke-linecap="round"/>
</svg>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 30px;'>Document Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Upload a document and let four AI agents analyze it together.</p>", unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"], label_visibility="collapsed")

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if uploaded_file is not None:
    if st.button("Run Analysis", type="primary"):
        st.session_state.qa_history = []  # clear old chat history for the new document

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Agents are analyzing the document... (Summarizer → Critic → Extractor)"):
            text = load_document(tmp_path)
            chunks = chunk_text(text)
            result = run_pipeline(text, chunks_for_qa=chunks)

        st.session_state.pipeline_result = result
        os.unlink(tmp_path)

if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    critique = result["critique_result"]
    score = critique.get("quality_score", "N/A")
    verdict = critique.get("verdict", "unknown")
    issues = critique.get("issues_found", [])

    # --- Sidebar: Quality Review + Extracted Data + Summary ---
    with st.sidebar:
        with st.expander("⚠️ Quality review notes"):
            if issues:
                for issue in issues:
                    st.markdown(f"""
                    <div style="background:#FAEEDA;border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                    <p style="font-size:14px;color:#633806;margin:0;">{issue}</p></div>
                    """, unsafe_allow_html=True)
                st.caption(critique.get("notes", ""))
            else:
                st.success("No issues found.")

        with st.expander("🔍 View extracted data"):
            st.json(result["extracted_data"])

        st.divider()

        st.markdown(f"""
        <p style="font-weight:500;font-size:17px;margin:0 0 12px;">📝 Summary</p>
        <p style="font-size:15px;color:#333;line-height:1.7;white-space:pre-wrap;">{result["summary"]}</p>
        """, unsafe_allow_html=True)

    # --- Main area: score cards ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="background:#FFF;border:1px solid #EEE;border-radius:12px;padding:16px;text-align:center;">
        <p style="font-size:12px;color:#888;margin:0 0 4px;">Quality score</p>
        <p style="font-size:22px;font-weight:500;margin:0;color:#1A1A1A;">{score}/10</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        verdict_bg = "#FBEAE6" if verdict == "needs_revision" else "#EAF3DE"
        verdict_border = "#F0C4B3" if verdict == "needs_revision" else "#C0DD97"
        verdict_text = "#712B13" if verdict == "needs_revision" else "#27500A"
        st.markdown(f"""
        <div style="background:{verdict_bg};border:1px solid {verdict_border};border-radius:12px;padding:16px;text-align:center;">
        <p style="font-size:12px;color:{verdict_text};margin:0 0 4px;">Verdict</p>
        <p style="font-size:15px;font-weight:500;margin:0;color:{verdict_text};">{verdict.replace('_', ' ').title()}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background:#FFF;border:1px solid #EEE;border-radius:12px;padding:16px;text-align:center;">
        <p style="font-size:12px;color:#888;margin:0 0 4px;">Issues found</p>
        <p style="font-size:22px;font-weight:500;margin:0;color:#1A1A1A;">{len(issues)}</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # --- Main area: chat ---
    st.markdown("<p style='font-weight:500;font-size:15px;margin:0 0 8px;color:#1A1A1A;'>💬 Ask about this document</p>", unsafe_allow_html=True)

    for q, a in st.session_state.qa_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    question = st.chat_input("Ask something about the document")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(question)
            st.write(answer)
        st.session_state.qa_history.append((question, answer))

else:
    st.info("Upload a document above and click 'Run Analysis' to get started.")