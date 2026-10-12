import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

google_api_key = os.getenv("GOOGLE_API_KEY")


load_dotenv()

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Agent with PDF Upload")
st.subheader("Upload your PDFs and ask questions about their content.")


# --------------------------------------------------
# 1. Initialize session state
# --------------------------------------------------

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "pdf_chat_session"

if "document_names" not in st.session_state:
    st.session_state.document_names = []


# --------------------------------------------------
# 2. Process uploaded PDF documents
# --------------------------------------------------

def process_document(path):

    # Load all PDFs from the specified directory
    loader = PyPDFDirectoryLoader(str(path))
    docs = loader.load()

    if not docs:
        raise ValueError("No readable PDF documents were found.")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splitted_data = splitter.split_documents(docs)

    if not splitted_data:
        raise ValueError("No text could be extracted from the PDFs.")

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )

    # Create vector store
    vector_store = InMemoryVectorStore.from_documents(
        documents=splitted_data,
        embedding=embeddings
    )

    # Initialize LLM
    llm = ChatGroq(
        model="openai/gpt-oss-120b"
    )

    # Define the retrieval tool
    @tool
    def retriever_tool(query: str) -> str:
        """Retrieve relevant information from the uploaded PDF documents."""

        retrieved_docs = vector_store.similarity_search(
            query=query,
            k=3
        )

        if not retrieved_docs:
            return "No relevant information was found in the documents."

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        return context

    # System prompt
    system_prompt = """
    You are a helpful assistant that answers questions using uploaded PDFs.

    For questions about the uploaded documents, use retriever_tool
    to retrieve relevant information before answering.

    Base your answers on the retrieved context.
    If the answer is not available in the retrieved context,
    clearly state that it could not be found in the uploaded documents.

    Do not invent facts.
    """

    # Create memory
    memory = InMemorySaver()

    # Create agent
    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    # Save everything in Streamlit session state
    st.session_state.agent = agent
    st.session_state.vector_store = vector_store
    st.session_state.document_uploaded = True
    st.session_state.messages = []
    st.session_state.thread_id = "pdf_chat_session"


# --------------------------------------------------
# 3. PDF upload interface
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process PDFs", disabled=not uploaded_files):

    try:
        with st.spinner("Processing PDFs and creating embeddings..."):

            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:

                pdf_dir = Path(temp_dir)

                # Save uploaded PDFs temporarily
                for uploaded_file in uploaded_files:

                    safe_name = Path(uploaded_file.name).name

                    file_path = pdf_dir / safe_name

                    file_path.write_bytes(
                        uploaded_file.getvalue()
                    )

                # Process PDFs before the temporary directory is removed
                process_document(pdf_dir)

            st.session_state.document_names = [
                file.name for file in uploaded_files
            ]

        st.success("PDFs processed successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Error processing PDFs: {e}")


# --------------------------------------------------
# 4. Display uploaded document information
# --------------------------------------------------

if st.session_state.document_uploaded:

    st.success("Your document knowledge base is ready.")

    with st.expander("View uploaded documents"):
        for name in st.session_state.document_names:
            st.write(f"📄 {name}")


# --------------------------------------------------
# 5. Chat interface
# --------------------------------------------------

if st.session_state.agent:

    st.subheader("Chat with your documents")

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    query = st.chat_input(
        "Ask anything about your uploaded PDFs..."
    )

    if query:

        # Display user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        # Generate ai response
        with st.chat_message("ai"):

            with st.spinner("Thinking..."):

                try:
                    response = st.session_state.agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": query
                                }
                            ]
                        },
                        config={
                            "configurable": {
                                "thread_id": st.session_state.thread_id
                            }
                        }
                    )

                    result = response["messages"][-1].content

                    st.markdown(result)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result
                        }
                    )

                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # Reset chat
    if st.button("Clear chat"):

        st.session_state.messages = []
        st.session_state.thread_id = (
            f"pdf_chat_{os.urandom(8).hex()}"
        )

        st.rerun()