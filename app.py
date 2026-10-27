import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
import tempfile



from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.title("RAG PDF CHATBOT") 
st.divider()
st.write("This is a chatbot that can answer the user's questions about a PDF document using RAG techniques.")


embeddings = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe",
    base_url="http://localhost:11434")
llm=ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0)
prompt_template=PromptTemplate.from_template("""
You are a helpful PDF question-answering assistant.

Answer the question using only the context provided below.
If the answer is not present in the context, clearly say
that you could not find the answer in the uploaded PDF.

Context:
{context}

Question:
{question}

Answer:
""")
parser=StrOutputParser()
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)   
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        st.success(f"Uploaded: {uploaded_file.name}")

    try:
        # Load the PDF using its local file path
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        st.success("PDF loaded successfully!")
        st.write(f"Total pages loaded: {len(documents)}")

    finally:
        # Delete the temporary file after loading
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts=text_splitter.split_documents(documents)
    vectorstore=FAISS.from_documents(texts, embeddings)
    chain={'context': vectorstore.as_retriever(), 'question': RunnablePassthrough()} | prompt_template | llm | parser
    question=st.text_input("Ask a question about the PDF document")
    res=chain.invoke(question)  
    st.write(res)
