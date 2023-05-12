from llms import GPT4AllJApi
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import argparse
import pickle
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

system_template = """Use the following pieces of context to answer the users question. 
If you don't know the answer, just say "Hmm..., I'm not sure.", don't try to make up an answer.
----------------
{context}
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


def get_llm():
    llm = GPT4AllJApi()
    return llm


def get_chain(store):
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(
        get_llm(),
        chain_type="stuff",
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    return chain


parser = argparse.ArgumentParser(description='FlutterGPT Q&A')
parser.add_argument('question', type=str, help='Your question for FlutterGPT')
args = parser.parse_args()

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)
chain = get_chain(store)
response = chain({"query": args.question})

print(f"Answer: {response['result']}")
print('\nSources:')
for source in response["source_documents"]:
    print(source.metadata['source'])
