## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

# --- STEP 1: Hugging Face API Token ---
if "hf_token" not in st.session_state:
    st.session_state.hf_token = None

api_key = st.text_input("Enter your Hugging Face API key:", type="password")
submit_token = st.button("Set Token & Load Model")

if submit_token and api_key:
    st.session_state.hf_token = api_key
    st.success("Token saved! You can now load the model.")

# --- STEP 2: Load Model & Embeddings ---
if st.session_state.hf_token and "llm_pipeline" not in st.session_state:
    os.environ["HF_TOKEN"] = st.session_state.hf_token
    st.session_state.llm_pipeline = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        task="text-generation"
    )
    st.session_state.llm = ChatHuggingFace(llm=st.session_state.llm_pipeline)
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    st.success("Model loaded successfully!")

# --- STEP 3: Session ID & History ---
if "store" not in st.session_state:
    st.session_state.store = {}

session_id = st.text_input("Session ID", value="default_session")

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# --- STEP 4: Upload PDFs ---
uploaded_files = st.file_uploader(
    "Choose PDF files", type="pdf", accept_multiple_files=True
)

documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        temp_pdf = f"./{uploaded_file.name}"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = uploaded_file.name

        documents.extend(docs)

# --- STEP 5: Create Vectorstore ---
if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=st.session_state.embeddings)
    retriever = vectorstore.as_retriever()

    # --- STEP 6: History-aware Retriever ---
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, retriever, contextualize_q_prompt
    )

    # --- STEP 7: QA Chain ---
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # --- STEP 8: User Input & Response ---
    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("Assistant:", response["answer"])
        st.write("Chat History:", session_history.messages)

else:
    st.info("Upload PDFs to start the conversation.")
