import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DEFAULT_URL = (
    "https://agritech.tnau.ac.in/"
    "agricultural_engineering/agriengg_govt_schemes.html"
)

st.set_page_config(
    page_title="TNAU Agricultural Schemes RAG",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 TNAU Agricultural Government Schemes – RAG")
st.caption(
    "Read the TNAU webpage directly, index its content in FAISS, "
    "and retrieve relevant passages."
)

def show_api_error(error: requests.RequestException):
    st.error(f"Could not communicate with FastAPI: {error}")
    response = getattr(error, "response", None)
    if response is not None:
        st.code(response.text)


tab_ingest, tab_search, tab_sources = st.tabs(
    ["🌐 Website ingestion", "🔎 Search FAISS", "📚 Index status"]
)

with tab_ingest:
    st.header("Pipeline 1: URL → extract → split → vectorize → FAISS")
    st.write(
        "The backend fetches the webpage, extracts readable text, "
        "splits it into chunks, creates local vectors and stores them in FAISS."
    )

    url = st.text_input("Website URL", value=DEFAULT_URL)
    st.info(
        "This starter uses scikit-learn HashingVectorizer. It works locally "
        "without downloading a Hugging Face model. It performs lexical vector "
        "retrieval, not neural semantic embeddings."
    )

    if st.button("Fetch website and save to FAISS", type="primary"):
        if not url.strip().startswith(("http://", "https://")):
            st.warning("Enter a valid http:// or https:// URL.")
        else:
            with st.spinner("Fetching webpage, extracting, splitting and indexing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ingest-url",
                        json={"url": url.strip()},
                        timeout=180,
                    )
                    response.raise_for_status()
                    data = response.json()

                    st.success(data.get("message", "Ingestion completed"))
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Characters extracted", data.get("characters_extracted", 0))
                    col2.metric("Chunks added", data.get("chunks_added", 0))
                    col3.metric("Total indexed chunks", data.get("total_chunks", 0))
                    st.json(data)

                except requests.RequestException as exc:
                    show_api_error(exc)

with tab_search:
    st.header("Pipeline 2: Question → FAISS → matching passages")
    question = st.text_input(
        "Ask a question about TNAU schemes",
        placeholder="What subsidy is available under Agricultural Mechanisation Programme?",
    )
    top_k = st.slider("Number of passages to retrieve", 1, 10, 5)

    if st.button("Search FAISS", type="primary"):
        if len(question.strip()) < 2:
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching FAISS..."):
                try:
                    response = requests.post(
                        f"{API_URL}/search",
                        json={"question": question.strip(), "top_k": top_k},
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()

                    if not data["results"]:
                        st.warning("No indexed passages found. Run Website ingestion first.")
                    else:
                        st.success(f"Retrieved {data['count']} passages.")
                        for number, item in enumerate(data["results"], start=1):
                            heading = (
                                f"Result {number} · similarity {item['score']:.3f} "
                                f"· chunk {item['chunk']}"
                            )
                            with st.expander(heading, expanded=(number == 1)):
                                st.write(item["text"])
                                st.caption(f"Source: {item['source']}")
                                st.caption(f"URL: {item['url']}")
                                st.caption(f"Fetched: {item['fetched_at']}")

                except requests.RequestException as exc:
                    show_api_error(exc)

with tab_sources:
    st.header("FAISS index and indexed sources")

    if st.button("Refresh index status"):
        st.rerun()

    try:
        stats_response = requests.get(f"{API_URL}/stats", timeout=30)
        stats_response.raise_for_status()
        stats = stats_response.json()

        c1, c2 = st.columns(2)
        c1.metric("Vectors in FAISS", stats.get("vector_count", 0))
        c2.metric("Metadata records", stats.get("metadata_count", 0))
        st.write("FAISS ready:", stats.get("faiss_ready", False))

        sources_response = requests.get(f"{API_URL}/sources", timeout=30)
        sources_response.raise_for_status()
        sources = sources_response.json().get("sources", [])

        if sources:
            st.dataframe(sources, use_container_width=True, hide_index=True)
        else:
            st.info("No sources indexed yet.")

    except requests.RequestException as exc:
        show_api_error(exc)

st.divider()
st.caption(
    "This version retrieves source passages; it does not yet generate "
    "LLM-written answers. Verify scheme details with the responsible department."
)
