import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Load the vector store
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create memory and QA chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=db.as_retriever(),
    memory=memory
)

# Test it
query = "What is gradient descent?"
result = qa({"question": query})
print(result["answer"])

query2 = "Explain it like I'm 5."  # Follow-up question
result2 = qa({"question": query2})  # Remembers previous chat
print(result2["answer"])
