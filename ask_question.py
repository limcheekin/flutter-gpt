import argparse
import faiss
import os
import pickle

from langchain.llms import HuggingFaceHub
from langchain.chains import VectorDBQAWithSourcesChain

parser = argparse.ArgumentParser(description='FlutterGPT Q&A')
parser.add_argument('question', type=str, help='Your question for FlutterGPT')
args = parser.parse_args()

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0})
chain = VectorDBQAWithSourcesChain.from_llm(llm=llm, vectorstore=store, verbose=True)
result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")

