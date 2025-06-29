import os
import re
import logging
from uuid import uuid4
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_docs(dir_path: str):
    """
    Read the documents.
    """
    logger.info(f"Starting to read documents from directory: {dir_path}")
    docs = []
    
    if not os.path.exists(dir_path):
        logger.error(f"Directory {dir_path} does not exist")
        return docs
    
    files = os.listdir(dir_path)
    logger.info(f"Found {len(files)} files in directory")
    
    for i, file in enumerate(files, 1):
        file_path = os.path.join(dir_path, file)
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read()
                doc = Document(page_content=content, metadata={"file_id": file})
                docs.append(doc)
                logger.info(f"Successfully read file {i}/{len(files)}: {file}")
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")
    
    logger.info(f"Successfully read {len(docs)} documents")
    return docs

def create_metadata(doc: Document) -> dict:
    """
    Create a metadata for a document.
    """
    logger.debug(f"Creating metadata for document: {doc.metadata.get('file_id', 'unknown')}")
    
    patterns = {
        'نام': r'نام:\s*(.+)',
        'دانشکده': r'دانشکده:\s*(.+)',
        'آدرس بخش': r'آدرس بخش:\s*(.+)',
        'آدرس شخصی': r'آدرس شخصی:\s*(.+)'
    }   
    content = doc.page_content
    file_metadata = {
        'file_id': doc.metadata.get("file_id", ""),
        'نام': None,
        'دانشکده': None,
        'آدرس بخش': None,
        'آدرس شخصی': None
    }
        
    extracted_fields = 0
    for field, pattern in patterns.items():
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            file_metadata[field] = match.group(1).strip()
            extracted_fields += 1
    
    logger.debug(f"Extracted {extracted_fields} metadata fields for {file_metadata['file_id']}")
    return file_metadata

def create_database(dir_path: str):
    """
    Create a vector database.
    """
    logger.info("Starting database creation process")
    
    # Read documents
    docs = read_docs(dir_path)
    if not docs:
        logger.error("No documents found. Exiting database creation.")
        return None
    
    # Create metadata for each document
    logger.info("Creating metadata for documents")
    for i, doc in enumerate(docs):
        metadata = create_metadata(doc)
        doc.metadata.update(metadata)
        doc.id = i
        if (i + 1) % 10 == 0 or i == len(docs) - 1:
            logger.info(f"Processed metadata for {i + 1}/{len(docs)} documents")

    # Generate UUIDs
    logger.info("Generating UUIDs for documents")
    uuids = [str(uuid4()) for _ in range(len(docs))]

    # Initialize embeddings
    logger.info("Initializing HuggingFace embeddings model")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="heydariAI/persian-embeddings")
        logger.info("Successfully loaded embeddings model")
    except Exception as e:
        logger.error(f"Failed to load embeddings model: {str(e)}")
        return None

    # Create vector store
    logger.info("Creating Chroma vector store")
    try:
        vectorstore = Chroma(
                collection_name="professors",
                embedding_function=embeddings,
                persist_directory="chroma_db"
        )
        logger.info("Successfully created Chroma vector store")
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        return None

    # Add documents to vector store
    logger.info(f"Adding {len(docs)} documents to vector store")
    try:
        vectorstore.add_documents(docs, ids=uuids)
        logger.info("Successfully added documents to vector store")
    except Exception as e:
        logger.error(f"Failed to add documents to vector store: {str(e)}")
        return None

    logger.info("Database creation completed successfully")
    return vectorstore

if __name__ == "__main__":
    logger.info("Starting database creation script")
    try:
        vectorstore = create_database("scraped_content")
        if vectorstore:
            logger.info("Script completed successfully")
        else:
            logger.error("Script failed to create database")
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
