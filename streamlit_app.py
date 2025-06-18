import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports - fixed for current versions
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Step 1: Create FAISS index if it doesn't exist
def create_faiss_index():
    from data_loader import generate_combined_file
    
    generate_combined_file()  # fetch + save data
    loader = TextLoader("data/combined.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    return db

# Step 2: Load or create vector store
if os.path.exists("faiss_index"):
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    db = create_faiss_index()

# Step 3: Set up memory and QA chain
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=db.as_retriever(),
    memory=st.session_state.memory
)

# Step 4: Streamlit UI
st.title("📘 Virtual TA (with Memory)")

user_input = st.text_input("Ask a question:")
if user_input:
    result = qa({"question": user_input})
    st.markdown("### 🧠 Answer:")
    st.write(result["answer"])

# Optional: Show chat history
if st.session_state.memory.buffer:
    st.markdown("---")
    st.markdown("### 💬 Chat History")
    for msg in st.session_state.memory.buffer:
        role = "👤 You" if msg.type == "human" else "🤖 TA"
        st.markdown(f"**{role}:** {msg.content}")
