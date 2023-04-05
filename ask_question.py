import argparse
import pickle

from langchain.llms import HuggingFacePipeline
from transformers import GPT2TokenizerFast, pipeline, set_seed
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain

system_template = """Use the following pieces of context to answer the users question. 
If you don't know the answer, just say "Hmm..., I'm not sure.", don't try to make up an answer.
ALWAYS return a "Sources" part in your answer.
The "Sources" part should be a reference to the source of the document from which you got your answer.
Example of your response should be:
```
The answer is foo
Sources:
1. abc
2. xyz
```
Begin!
----------------
{summaries}
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


def get_llm():
    model_id = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        max_length=1024
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
