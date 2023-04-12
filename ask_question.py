import argparse
import pickle

from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, set_seed
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain

print('loading model...')

try:
    model = AutoModelForSeq2SeqLM.from_pretrained("flan-t5-base/")
    tokenizer = AutoTokenizer.from_pretrained("flan-t5-base/")
except:
    print("An exception occurred on loading model.")

system_template = """Use the following pieces of context to answer the users question. 
If you don't know the answer, just say "Hmm..., I'm not sure.", don't try to make up an answer.
----------------
{summaries}
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        min_length=128,
        max_length=256,
        temperature=0.0,
        use_fast=True,
        length_penalty=2,
        num_beams=4,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        early_stopping=True,
    )
    set_seed(55)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def get_chain(store):
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        get_llm(),
        chain_type="stuff",
        retriever=store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True,
        max_tokens_limit=512
    )
    return chain


parser = argparse.ArgumentParser(description='FlutterGPT Q&A')
parser.add_argument('question', type=str, help='Your question for FlutterGPT')
args = parser.parse_args()

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)
chain = get_chain(store)
result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
