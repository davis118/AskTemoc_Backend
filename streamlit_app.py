import streamlit as st
import requests
import json
import pandas as pd
import os

# ---------- Page config ----------
st.set_page_config(page_title="Document Manager", layout="wide", initial_sidebar_state="expanded")

# ---------- Initialize session state ----------
if "api_port" not in st.session_state:
    st.session_state.api_port = "8000"

if "api_base" not in st.session_state:
    st.session_state.api_base = f"http://localhost:{st.session_state.api_port}"

# ---------- Sidebar ----------
st.sidebar.title("Document Manager")
st.sidebar.markdown("---")

new_port = st.sidebar.text_input(
    "Backend Port",
    value=st.session_state.api_port,
    help="Port where the FastAPI backend is running, e.g., 8001"
)
if new_port != st.session_state.api_port:
    st.session_state.api_port = new_port
    st.session_state.api_base = f"http://localhost:{new_port}"
    st.rerun()

st.sidebar.info(f"Using API: {st.session_state.api_base}")

if st.sidebar.button("Test Connection"):
    try:
        r = requests.get(f"{st.session_state.api_base}/", timeout=3)
        if r.status_code == 200:
            st.sidebar.success("Connection successful")
        else:
            st.sidebar.error(f"Error {r.status_code}")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Dashboard", "Upload Document", "Manage Documents", "Document Details"])
st.sidebar.markdown("---")

# ---------- Helper functions ----------
def api_call(method, endpoint, **kwargs):
    url = f"{st.session_state.api_base}{endpoint}"
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
        return None

def list_documents():
    return api_call("GET", "/documents/")

def delete_document(doc_id, hard=False):
    return api_call("DELETE", f"/documents/{doc_id}", params={"hard_delete": hard})

def upload_file(file, title, metadata):
    files = {"file": file}
    data = {}
    if title:
        data["title"] = title
    if metadata:
        data["metadata"] = json.dumps(metadata)
    return api_call("POST", "/documents/upload/file", files=files, data=data)

def upload_html(html, source_url, title, metadata):
    data = {"html": html}
    if source_url:
        data["source_url"] = source_url
    if title:
        data["title"] = title
    if metadata:
        data["metadata"] = json.dumps(metadata)
    return api_call("POST", "/documents/upload/html", data=data)

def upload_url(url, metadata):
    data = {"url": url}
    if metadata:
        data["metadata"] = json.dumps(metadata)
    return api_call("POST", "/documents/upload/url", data=data)

def update_document(doc_id, title, metadata):
    data = {}
    if title:
        data["title"] = title
    if metadata:
        data["metadata"] = json.dumps(metadata)
    return api_call("PUT", f"/documents/{doc_id}", data=data)

def get_document_details(doc_id):
    return api_call("GET", f"/documents/{doc_id}")

# ---------- Pages ----------
if page == "Dashboard":
    st.header("Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recent Documents")
        docs = list_documents()
        if docs and "documents" in docs:
            df = pd.DataFrame(docs["documents"])
            if not df.empty:
                df_display = df[["id", "title", "source", "created_at", "is_deleted"]]
                # Use width='stretch' for full container width and add vertical scroll
                st.dataframe(df_display, width='stretch', height=400)
            else:
                st.info("No documents found.")
        else:
            st.info("No documents found.")
    with col2:
        st.subheader("Quick Stats")
        if docs and "documents" in docs:
            total = len(docs["documents"])
            active = sum(1 for d in docs["documents"] if not d["is_deleted"])
            st.metric("Total Documents", total)
            st.metric("Active Documents", active)

elif page == "Upload Document":
    st.header("Upload New Document")
    tab1, tab2, tab3 = st.tabs(["File Upload", "URL", "HTML"])

    with tab1:
        with st.form("upload_file"):
            uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "md", "html"])
            title = st.text_input("Document Title (optional)")
            metadata = st.text_area("Metadata (JSON)", height=100, placeholder='{"author": "John", "tags": ["report"]}')
            submitted = st.form_submit_button("Upload")
            if submitted and uploaded_file:
                try:
                    meta = json.loads(metadata) if metadata else {}
                except json.JSONDecodeError:
                    st.error("Invalid JSON in metadata")
                else:
                    with st.spinner("Uploading and processing..."):
                        result = upload_file(uploaded_file, title, meta)
                    if result:
                        st.success(f"Document uploaded: {result['document']['title']}")
                        st.json(result["document"])

    with tab2:
        with st.form("upload_url"):
            url = st.text_input("URL")
            metadata = st.text_area("Metadata (JSON)", height=100, placeholder='{"source": "web"}')
            submitted = st.form_submit_button("Fetch and Ingest")
            if submitted and url:
                try:
                    meta = json.loads(metadata) if metadata else {}
                except json.JSONDecodeError:
                    st.error("Invalid JSON in metadata")
                else:
                    with st.spinner("Fetching URL and processing..."):
                        result = upload_url(url, meta)
                    if result:
                        st.success(f"Document ingested from URL: {result['document']['title']}")
                        st.json(result["document"])

    with tab3:
        with st.form("upload_html"):
            html_content = st.text_area("HTML Content", height=200)
            source_url = st.text_input("Source URL (optional)")
            title = st.text_input("Document Title (optional)")
            metadata = st.text_area("Metadata (JSON)", height=100)
            submitted = st.form_submit_button("Ingest HTML")
            if submitted and html_content:
                try:
                    meta = json.loads(metadata) if metadata else {}
                except json.JSONDecodeError:
                    st.error("Invalid JSON in metadata")
                else:
                    with st.spinner("Processing HTML..."):
                        result = upload_html(html_content, source_url, title, meta)
                    if result:
                        st.success(f"HTML ingested: {result['document']['title']}")
                        st.json(result["document"])

elif page == "Manage Documents":
    st.header("Manage Documents")
    docs = list_documents()
    if not docs or "documents" not in docs or not docs["documents"]:
        st.info("No documents found.")
    else:
        for doc in docs["documents"]:
            with st.expander(f"{doc['title']} (ID: {doc['id'][:8]}...)"):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**Source:** {doc['source']}")
                    st.write(f"**Created:** {doc['created_at']}")
                    st.write(f"**Deleted:** {doc['is_deleted']}")
                with col2:
                    if doc["is_deleted"]:
                        if st.button("Restore", key=f"restore_{doc['id']}"):
                            st.warning("Restore not implemented yet.")
                    else:
                        if st.button("Soft Delete", key=f"soft_{doc['id']}"):
                            result = delete_document(doc["id"], hard=False)
                            if result:
                                st.success(f"Document {doc['title']} soft-deleted.")
                                st.rerun()
                with col3:
                    if st.button("Hard Delete", key=f"hard_{doc['id']}", type="primary"):
                        if st.checkbox(f"Confirm hard delete of '{doc['title']}'?", key=f"confirm_{doc['id']}"):
                            result = delete_document(doc["id"], hard=True)
                            if result:
                                st.success(f"Document {doc['title']} permanently deleted.")
                                st.rerun()
                            else:
                                st.error("Delete failed.")
                if doc.get("doc_metadata"):
                    st.json(doc["doc_metadata"], expanded=False)

elif page == "Document Details":
    st.header("Document Details")
    doc_id = st.text_input("Document ID")
    if doc_id:
        with st.spinner("Fetching details..."):
            data = get_document_details(doc_id)
        if data:
            doc = data["document"]
            chunks = data.get("chunks", [])
            st.subheader(doc["title"])
            st.write(f"**ID:** {doc['id']}")
            st.write(f"**Source:** {doc['source']}")
            st.write(f"**Created:** {doc['created_at']}")
            st.write(f"**Updated:** {doc['updated_at']}")
            st.write(f"**Deleted:** {doc['is_deleted']}")
            if doc.get("doc_metadata"):
                st.subheader("Metadata")
                st.json(doc["doc_metadata"])

            st.subheader(f"Chunks ({len(chunks)})")
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {chunk['chunk_index']} (ID: {chunk['id'][:8]}...)"):
                    st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
                    if chunk.get("chunk_metadata"):
                        st.caption("Metadata:")
                        st.json(chunk["chunk_metadata"], expanded=False)
        else:
            st.error("Document not found")