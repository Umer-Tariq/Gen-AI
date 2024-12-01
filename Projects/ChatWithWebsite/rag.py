import streamlit as st
import os
import shutil
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain_community.document_loaders import WebBaseLoader

def get_unique_persist_dir():
    """Generate a unique directory for each vector store session."""
    base_dir = "./chroma_db"
    unique_dir = os.path.join(base_dir, str(uuid.uuid4()))
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

def cleanup_old_vectorstores(base_dir="./chroma_db", keep_recent=3):
    """
    Clean up old Chroma vector store directories, keeping the most recent ones.
    
    Args:
        base_dir (str): Base directory containing vector store directories
        keep_recent (int): Number of recent directories to keep
    """
    try:
        # List all subdirectories sorted by creation time (newest first)
        if not os.path.exists(base_dir):
            return
        
        dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
        
        # Sort directories by creation time (newest first)
        dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
        
        # Remove older directories
        for old_dir in dirs[keep_recent:]:
            try:
                shutil.rmtree(old_dir)
            except Exception as e:
                st.warning(f"Could not remove {old_dir}: {e}")
    except Exception as e:
        st.error(f"Error in cleanup: {e}")

def get_answer(url, question):
    try:
        # Generate a unique persist directory
        persist_dir = get_unique_persist_dir()
        
        # Cleanup old vector stores
        cleanup_old_vectorstores()

        # Step 1: Load the website content using WebBaseLoader
        loader = WebBaseLoader(web_paths=[url])
        docs = loader.load()

        # Step 2: Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Step 3: Create vector store with unique persistent directory
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
            persist_directory=persist_dir
        )

        # Persist the database
        vectorstore.persist()

        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # Step 4: Set up the LLM and prompt template
        llm = ChatMistralAI()
        rag_prompt_template = """
        You are an assistant that is helping a user interact with content from a specific website. The content of the website has been extracted and is provided below.

        Content:
        "{context}"

        The user has a question related to the content. Your task is to provide an accurate and helpful answer based on the context provided above.

        User Question:
        "{question}"

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(rag_prompt_template)

        # Step 5: Create RAG Chain
        rag_chain = (
            RunnableMap(
                {"context": itemgetter("question") | retriever,
                 "question": itemgetter("question")}
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        # Step 6: Invoke the chain and get the answer
        response = rag_chain.invoke({"question": question})
        return response

    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI
def main():
    st.title("Chat with a Website")
    st.write("Enter a URL and ask a question based on the content of that website!")

    # Ensure base directory exists
    os.makedirs("./chroma_db", exist_ok=True)

    # Input fields for the URL and Question
    url = st.text_input("Enter URL of the Website:")
    question = st.text_input("Enter your Question:")

    # If both inputs are provided, start processing
    if st.button("Get Answer") and url and question:
        with st.spinner('Processing...'):
            # Directly call the function to ensure fresh processing each time
            answer = get_answer(url, question)
        st.success("Answer:")
        st.write(answer)

# Run the app
if __name__ == "__main__":
    main()