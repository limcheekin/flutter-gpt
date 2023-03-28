import argparse
import os

from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient

parser = argparse.ArgumentParser(description='FlutterGPT Q&A')
parser.add_argument('question', type=str, help='Your question for FlutterGPT')
args = parser.parse_args()

url = os.environ.get("QDRANT_URL")
api_key = os.environ.get("QDRANT_API_KEY")
qdrant = Qdrant(QdrantClient(url=url, api_key=api_key), "docs_flutter_dev", embedding_function=OpenAIEmbeddings().embed_query)
chain = VectorDBQAWithSourcesChain.from_llm(
        llm=OpenAI(temperature=0, verbose=True), vectorstore=qdrant, verbose=True)
result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")

