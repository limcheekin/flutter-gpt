from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

def ingest_data():
    file_paths = None
    with open('html_files_index.txt', 'r') as file: 
        file_paths = file.readlines()

    docs = []
    text_splitter = RecursiveCharacterTextSplitter()

    print("Load HTML files locally...")
    for i, file_path in enumerate(file_paths):
        file_path = file_path.rstrip("\n")
        doc = UnstructuredHTMLLoader(file_path).load()
        splits = text_splitter.split_documents(doc)
        docs.extend(splits)
        print(f"{i+1})Split {file_path} into {len(splits)} chunks")

    print("Load data to FAISS store")
    store = FAISS.from_documents(docs, OpenAIEmbeddings())

    print("Save faiss_store.pkl")
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
        
if __name__ == "__main__":
    ingest_data()

