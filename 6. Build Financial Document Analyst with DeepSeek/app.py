import streamlit as st
from rag import load_and_convert_document, get_markdown_splits, create_or_load_vector_store, build_rag_chain
from langchain_ollama import OllamaEmbeddings
from pathlib import Path

# Streamlit heading
st.title("Build RAG Locally with DeepSeek for Financial Data Analysis")

# File upload for the PDF
uploaded_file = st.file_uploader("Upload a PDF file for analysis", type=["pdf"])

# Input question field
question = st.text_input("Enter your question:", placeholder="e.g., What is the company's revenue for the quarter?")

# Button to process and generate answers
if st.button("Submit") and uploaded_file and question:
    with st.spinner("Processing document..."):
        # Load and process the document
        filename = uploaded_file.name.split(".")[0]
        pdf_content = uploaded_file.read()

        # Save the file temporarily
        temp_path = f"temp_{filename}.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf_content)

        markdown_content = load_and_convert_document(temp_path)
        chunks = get_markdown_splits(markdown_content)

        # Initialize embeddings
        embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

        # Create or load vector DB
        vector_store = create_or_load_vector_store(filename, chunks, embeddings)

        # Build retriever
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})

        # Build and run the RAG chain
        rag_chain = build_rag_chain(retriever)

        # Get response from the RAG chain
        with st.spinner("Answering your question..."):
            response = "".join(rag_chain.stream(question))

        # Display the response
        st.subheader("Answer:")
        st.markdown(response)

        # Clean up temporary file
        Path(temp_path).unlink()
