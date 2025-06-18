import os
import dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain.chains.retrieval import create_retrieval_chain

print("Starting application...")

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

    print("Setting up retriever...")
    retriever = vectorstore.as_retriever(search_type="mmr",
                                        search_kwargs={"fetch_k": 10, "k": 5})
    print("Retriever configured successfully")

    print("Initializing LLM...")
    llm = ChatOpenAI(
        base_url="https://api.avalai.ir/v1",
        model="gpt-4.1",
        temperature=0,
        api_key=api_key
    )
    print("LLM initialized successfully")

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
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    print("Creating question-answer chain...")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    print("Chain created successfully")

    print("Processing query...")
    response = chain.invoke({"input": "استاد جهانگرد کیست؟"})

    print("-"*100)
    print("Resources: ", response["context"])
    print("-"*100)
    print("Content: ", response["answer"])
    print("-"*100)

except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise
