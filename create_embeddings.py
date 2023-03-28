from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
import os 

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

    print("Load data to QDRANT")
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    qdrant = Qdrant.from_documents(docs, OpenAIEmbeddings(), url=url, api_key=api_key, 
                                   collection_name="docs_flutter_dev", prefer_grpc=True)

if __name__ == "__main__":
    ingest_data()

