from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import hashlib

tokenizer = AutoTokenizer.from_pretrained("bart-lfqa/")
md5 = hashlib.md5()


def get_doc_id(doc):
    md5.update(doc.metadata['source'].encode('utf-8'))
    uid = md5.hexdigest()[:12]
    # print(doc.metadata['source'])
    return uid


def token_len(text):  # token length function
    tokens = tokenizer.encode(text, max_length=1024, truncation=True)
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
        chunk_size=250,
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

    print("Load data to FAISS store")
    embeddings = HuggingFaceEmbeddings(
        model_name="flax-sentence-embeddings/all_datasets_v3_mpnet-base")
    faiss_db = FAISS.from_texts(texts, embeddings, metadatas=docs)
    print("Saving faiss.db")
    faiss_db.save_local("faiss.db")


if __name__ == "__main__":
    ingest_data()
