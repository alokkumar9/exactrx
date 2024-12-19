from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
import wikipediaapi
from langchain.prompts import PromptTemplate

#Get the wikipedia page text
def wikipedia_text_loader(page_name):
    # Define your custom User-Agent
    user_agent = "MyApp/1.0 (http://example.com; myemail@example.com)"
    # Initialize the Wikipedia object with the custom User-Agent
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=user_agent
    )
    # Retrieve a Wikipedia page
    page_py = wiki_wiki.page(page_name)

    # Print the content of the page
    return page_py.text


def retrieve_combiner(docs):
    combined_doc = "\n\n".join([doc.page_content for doc in docs])
    return combined_doc

#Check if the collection already exists
def check_collection_exists(collection_name: str, persist_directory: str = "./chroma_db") -> bool:
    chroma_client = Chroma(persist_directory=persist_directory)
    try:
        chroma_client._client.get_collection(collection_name)
        print("*********collection exists*********")
        return True
    except Exception as e:
        if "************Collection does not exist*********" in str(e):
            return False

#Stores the embedding in ChromaDB vector Database
def create_and_store_embedding(wiki_documents, collection_name, embedding_model):
    Chroma.from_documents(
        documents=wiki_documents,
        embedding=embedding_model,
        persist_directory="./chroma_db",
        collection_name=collection_name
    )

def load_collection_get_retriever(collection_name, embedding_model):
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory="./chroma_db",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return retriever

# Initialize models and components
embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en-v1.5')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
llm = OllamaLLM(model="llama3.2", temperature=0)

# Load and process Wikipedia documents

page_text=wikipedia_text_loader('Munich')
docs=text_splitter.create_documents(texts=[page_text])
# Check if collection exists, create if not
if not check_collection_exists("wiki_collection"):
    create_and_store_embedding(docs, "wiki_collection", embedding_model)

# Set up retriever and RAG chain
retriever = load_collection_get_retriever("wiki_collection", embedding_model)
# prompt = hub.pull("rlm/rag-prompt")



prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Following is the context:
    Context: {context}\n
    Question: {question}

    Answer:""",
    input_variables=["question", "context"]
)

#chain defined
wiki_rag_chain = (
    {"context": retriever | retrieve_combiner, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_input = str(input("ASK>> "))
print("=============RESPONSE====================")
for chunk in wiki_rag_chain.stream(user_input):
    print(chunk, end="", flush=True)



