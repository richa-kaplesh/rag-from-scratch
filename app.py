import streamlit as st
import requests

st.title("RAG from Scratch 📄")
st.caption("Upload a PDF and ask questions about it")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Ask a question about the document")

if st.button("Ask"):
    if not uploaded_file:
        st.warning("Please upload a PDF first")
    elif not query:
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                params={"query": query}
            )
            if response.status_code == 200:
                st.success("Answer:")
                st.write(response.json()["answer"])
            else:
                st.error(f"Error: {response.status_code}")