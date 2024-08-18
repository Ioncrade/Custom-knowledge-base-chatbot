import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pickle
import PyPDF2

load_dotenv()
CACHE_FILE = "txt_cache.pkl"
TEXT_FILES_DIR = "F:\programming\Chatbot"  # Specify the directory containing your text files
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_text_files(directory):
    """Load all text files from a specified directory."""
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text += file.read()
        elif filename.endswith(".pdf"):
            with open(directory, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page].extract_text()        
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def preprocess_and_store_text_files(directory):
    raw_text = load_text_files(directory)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    with open(CACHE_FILE, "wb") as cache_file:
        pickle.dump((raw_text, text_chunks), cache_file)

def load_cached_data():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as cache_file:
            raw_text, text_chunks = pickle.load(cache_file)
        return raw_text, text_chunks
    return None, None

def main():
    st.set_page_config(page_title="Knowledge Base Chatbot")
    st.header("Chat with the Knowledge Base")

    # Automatically process the text files in the specified directory
    raw_text, text_chunks = load_cached_data()
    if not raw_text or not text_chunks:
        with st.spinner("Processing knowledge base..."):
            preprocess_and_store_text_files(TEXT_FILES_DIR)
            st.success("Knowledge base created and stored.")

    user_question = st.text_input("Ask a question")

    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
