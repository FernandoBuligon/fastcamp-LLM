from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def url_to_vector(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    embeddings = OllamaEmbeddings(model="mistral")
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    retriever = vector.as_retriever()
    return retriever
