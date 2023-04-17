import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import hashlib
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()

# PGVector needs the connection string to the database.
# We will load it from the environment variables.
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("DB_DRIVER", "psycopg2"),
    host=os.environ.get("DB_HOST", "localhost"),
    port=int(os.environ.get("DB_PORT", "5432")),
    database=os.environ.get("DB_NAME", "postgres"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", "postgres"),
)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
md5 = hashlib.md5()


def get_doc_id(doc):
    md5.update(doc.metadata['source'].encode('utf-8'))
    uid = md5.hexdigest()[:12]
    # print(doc.metadata['source'])
    return uid


def token_len(text):  # token length function
    tokens = tokenizer.encode(text, max_length=512, truncation=True)
    return len(tokens)


def load_documents():
    file_paths = None
    with open('html_files_index.txt', 'r') as file:
        file_paths = file.readlines()

    docs = []
    for file_path in tqdm(file_paths):
        file_path = file_path.rstrip("\n")
        doc = UnstructuredHTMLLoader(file_path).load()
        doc[0].metadata['source'] = doc[0].metadata['source'].replace(
            './site', 'https://docs.flutter.dev')
        docs.extend(doc)

    print(f"len(docs) {len(docs)}")

    return docs


def split_documents(docs):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,  # number of tokens overlap between chunks
        length_function=token_len,
        separators=['\n\n', '\n', ' ', '']
    )
    for doc in tqdm(docs):
        id = get_doc_id(doc)
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            documents.append({
                'id': f'{id}-{i}',
                'text': chunk,
                'source': doc.metadata['source']
            })

    print(f"len(documents) {len(documents)}")
    return documents


def ingest_data():
    docs = load_documents()
    docs = split_documents(docs)
    # extract text from docs and id, source become metadata
    texts = [doc.pop('text') for doc in docs]
    ids = [doc.pop('id') for doc in docs]
    metadatas = docs

    print("Load data to PGVector store")
    embedding = HuggingFaceEmbeddings()
    db = PGVector.from_texts(texts, embedding, metadatas=metadatas, ids=ids,
                             collection_name="flutter_gpt",
                             connection_string=CONNECTION_STRING,
                             pre_delete_collection=True)
    query = "What is Flutter?"
    docs_with_score: List[Tuple[Document, float]
                          ] = db.similarity_search_with_score(query)

    for doc, score in docs_with_score:
        print("-" * 80)
        print("Score: ", score)
        print(doc.page_content)
        print("-" * 80)


if __name__ == "__main__":
    ingest_data()
