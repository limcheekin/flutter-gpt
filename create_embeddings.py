from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import GPT2TokenizerFast
import pickle


def ingest_data():
    file_paths = None
    with open('html_files_index.txt', 'r') as file:
        file_paths = file.readlines()

    docs = []
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=512, chunk_overlap=0)

    print("Load HTML files locally...")
    for i, file_path in enumerate(file_paths):
        file_path = file_path.rstrip("\n")
        doc = UnstructuredHTMLLoader(file_path).load()
        splits = text_splitter.split_documents(doc)
        docs.extend(splits)
        print(f"{i+1})Split {file_path} into {len(splits)} chunks")

    print("Load data to FAISS store")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    store = FAISS.from_documents(
        docs, HuggingFaceEmbeddings(model_name=model_name))

    print("Save faiss_store.pkl")
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)


if __name__ == "__main__":
    ingest_data()
