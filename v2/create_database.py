import os
import dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader

def create_vector_store():
    print("Starting database creation...")
    
    # Load environment variables
    dotenv.load_dotenv()
    api_key = os.getenv("AVALAI_API_KEY")
    if not api_key:
        raise ValueError("AVALAI_API_KEY not found in environment variables")
    
    os.environ["AVALAI_API_KEY"] = api_key
    print("Environment variables loaded successfully")
    
    try:
        print("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="heydariAI/persian-embeddings")
        print("Embeddings initialized successfully")
        
        # Check if vector store already exists
        if os.path.exists("./db") and os.path.isdir("./db"):
            print("Loading existing vector store from ./db")
            vectorstore = Chroma(
                persist_directory="./db",
                embedding_function=embeddings,
                collection_name="professors"
            )
            print("Vector store loaded successfully")
        else:
            print("Creating new vector store...")
            docs = []
            if not os.path.exists("scraped_content"):
                raise FileNotFoundError("scraped_content directory not found")
                
            for file in os.listdir("scraped_content"):
                if file.endswith(".txt"):
                    print(f"Loading document: {file}")
                    loader = TextLoader(f"scraped_content/{file}")
                    docs.extend(loader.load())
            
            print(f"Total documents loaded: {len(docs)}")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            
            vectorstore = Chroma.from_documents(
                documents=text_splitter.split_documents(docs),
                embedding=embeddings,
                collection_name="professors",
                persist_directory="./db"
            )
            print("New vector store created and saved")
        
        return vectorstore
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    create_vector_store()