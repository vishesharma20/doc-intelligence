import streamlit as st
import tempfile
import os

from app.loaders import load_document, chunk_text
from app.orchestrator import run_pipeline
from app.agents.qa import ask

st.set_page_config(page_title="Document Intelligence System", layout="wide")

st.title("📄 Multi-Agent Document Intelligence System")
st.caption("Upload a document and watch multiple AI agents analyze it collaboratively.")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

if uploaded_file is not None:
    if st.button("Run Analysis", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Agents are analyzing the document... (Summarizer → Critic → Extractor)"):
            text = load_document(tmp_path)
            chunks = chunk_text(text)
            result = run_pipeline(text, chunks_for_qa=chunks)

        st.session_state.pipeline_result = result
        st.session_state.chunks = chunks
        os.unlink(tmp_path)

if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔍 Extracted Data", "✅ Quality Review", "💬 Ask Questions"])

    with tab1:
        st.markdown(result["summary"])

    with tab2:
        st.json(result["extracted_data"])

    with tab3:
        critique = result["critique_result"]
        score = critique.get("quality_score", "N/A")
        verdict = critique.get("verdict", "unknown")

        col1, col2 = st.columns(2)
        col1.metric("Quality Score", f"{score}/10")
        col2.metric("Verdict", verdict.upper())

        st.subheader("Issues Found")
        issues = critique.get("issues_found", [])
        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.success("No issues found.")

        st.caption(critique.get("notes", ""))

    with tab4:
        question = st.text_input("Ask a question about the document")
        if question:
            with st.spinner("Retrieving answer..."):
                answer = ask(question)
            st.markdown(f"**Answer:** {answer}")
else:
    st.info("Upload a document above and click 'Run Analysis' to get started.")