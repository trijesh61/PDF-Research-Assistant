
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="PDF Research Assistant", page_icon="📚")

st.title("📚 PDF Research Assistant")
st.write("Upload your PDFs and ask questions about their content.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "files" not in st.session_state:
    st.session_state.files = []

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Remove documents"):
        st.session_state.vectorstore = None
        st.session_state.files = []
        st.session_state.messages = []
        st.rerun()

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files and len(uploaded_files) > 5:
    st.warning("You can upload a maximum of 5 PDFs.")
    st.stop()

if st.button("Process PDFs", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Reading your documents..."):
            try:
                documents = []

                for file in uploaded_files:
                    reader = PdfReader(BytesIO(file.getvalue()))

                    for page_number, page in enumerate(reader.pages, start=1):
                        text = page.extract_text() or ""

                        if text.strip():
                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": file.name,
                                        "page": page_number,
                                    },
                                )
                            )

                if not documents:
                    st.error("No readable text found in the uploaded PDFs.")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                )

                chunks = splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                )

                st.session_state.files = [file.name for file in uploaded_files]
                st.session_state.messages = []

                st.success(
                    f"Processed {len(uploaded_files)} PDFs successfully!"
                )

            except Exception as e:
                st.error(f"Something went wrong: {e}")

if st.session_state.files:
    st.subheader("Your documents")

    for filename in st.session_state.files:
        st.write(f"📄 {filename}")

if st.session_state.vectorstore:
    st.divider()
    st.subheader("Ask your questions")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View sources"):
                    for source in message["sources"]:
                        st.write(
                            f"{source['file']} — Page {source['page']}"
                        )

    question = st.chat_input("Ask something about your PDFs...")

    if question:
        if not api_key:
            st.warning("Please enter your Google API key in the sidebar.")
            st.stop()

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Finding an answer..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )

                    docs = retriever.invoke(question)

                    context = "\n\n".join(
                        f"File: {doc.metadata['source']}, "
                        f"Page: {doc.metadata['page']}\n"
                        f"{doc.page_content}"
                        for doc in docs
                    )

                    prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            """Answer the question using only the
                            information in the provided PDF context.
                            If the answer is not available, say so.
                            Include the relevant PDF filename and page
                            number in your answer.

                            Context:
                            {context}""",
                        ),
                        ("human", "{question}"),
                    ])

                    model = ChatGoogleGenerativeAI(
                        model="gemini-3.5-flash",
                        google_api_key=api_key,
                        temperature=0,
                    )

                    response = (prompt | model).invoke({
                        "context": context,
                        "question": question,
                    })

                    answer = response.content

                    if isinstance(answer, list):
                        answer = "\n".join(
                            part.get("text", str(part))
                            if isinstance(part, dict)
                            else str(part)
                            for part in answer
                        )

                    st.markdown(answer)

                    sources = []
                    seen = set()

                    for doc in docs:
                        source = (
                            doc.metadata["source"],
                            doc.metadata["page"],
                        )

                        if source not in seen:
                            seen.add(source)
                            sources.append({
                                "file": source[0],
                                "page": source[1],
                            })

                    if sources:
                        with st.expander("View sources"):
                            for source in sources:
                                st.write(
                                    f"{source['file']} — "
                                    f"Page {source['page']}"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"Couldn't generate an answer: {e}")
