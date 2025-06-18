# Import required libraries
import os
import dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

# Initialize FastAPI application
app = FastAPI()

# Load environment variables and get API key
dotenv.load_dotenv()
api_key = os.getenv("AVALAI_API_KEY")
if not api_key:
    raise ValueError("AVALAI_API_KEY not found in environment variables")

os.environ["AVALAI_API_KEY"] = api_key

# Initialize Persian language embeddings model
embeddings = HuggingFaceEmbeddings(model_name="heydariAI/persian-embeddings")

# Set up Chroma vector store for document storage and retrieval
vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=embeddings,
    collection_name="professors"
)

# Configure the retriever with MMR (Maximal Marginal Relevance) search
# fetch_k: number of documents to fetch initially
# k: number of documents to return after reranking
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 10, "k": 5})

compression_retriever = ContextualCompressionRetriever(
    base_compressor=FlashrankRerank(top_n=4),
    base_retriever=retriever
)

# Initialize the language model with AvalAI's API
llm = ChatOpenAI(
    base_url="https://api.avalai.ir/v1",
    model="gpt-4.1",
    temperature=0,
    api_key=api_key
)

# Define system prompt in Persian for the university assistant
# This prompt provides guidelines for responding to queries about professors
system_prompt = """
:شما یک دستیار هوشمند برای دانشگاه یزد هستید. وظیفه شما پاسخگویی به سوالات در مورد اساتید، برنامه کلاسی، سوابق تحصیلی و تخصص‌های آنها براساس انتخاب هوشمندانه از بافتار داده شده است.
بافتار: {context}

راهنمایی‌های مهم:
۱. پاسخ‌های خود را به زبان فارسی و با لحن رسمی و محترمانه ارائه دهید
۲. اطلاعات دقیق از جمله ساعات کلاس، شماره کلاس و جزئیات تماس را به طور دقیق ذکر کنید
۳. اگر اطلاعاتی در مورد موضوعی ندارید، صادقانه اعلام کنید
۴. در مورد زمینه‌های تخصصی و پژوهشی اساتید با دقت و جزئیات پاسخ دهید
۵. برنامه کلاسی را با ذکر روز، ساعت و نام درس به طور منظم ارائه دهید
"""

# Create chat prompt template with system and user messages
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Set up the question-answering chain using the LLM and prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# Create the final retrieval chain that combines document retrieval and question answering
chain = create_retrieval_chain(compression_retriever, question_answer_chain)

# Define the request model for query endpoint
class QueryRequest(BaseModel):
    query: str

# API endpoint for processing queries
@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        # Process the query through the chain and get response
        response = chain.invoke({"input": request.query})
        return {
            "answer": response["answer"],
            "context": [doc.page_content for doc in response["context"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)