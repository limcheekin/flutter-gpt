import argparse

from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

model = AutoModelForSeq2SeqLM.from_pretrained("bart-lfqa/")
tokenizer = AutoTokenizer.from_pretrained("bart-lfqa/")

system_template = """Answer the question with the following context below. 
{summaries}
If you don't know the answer, just say "Hmm..., I'm not sure.".
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


def get_llm():
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        min_length=32,
        max_length=256,
        temperature=0.0,
        do_sample=False,
        early_stopping=True,
        num_beams=8,
        top_k=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=1
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm


def get_chain(vector_store):
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        get_llm(),
        chain_type="stuff",
        verbose=True,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True,
        max_tokens_limit=1024,
        return_source_documents=True,
    )
    return chain


parser = argparse.ArgumentParser(description='FlutterGPT Q&A')
parser.add_argument('question', type=str, help='Your question for FlutterGPT')
args = parser.parse_args()

embeddings = HuggingFaceEmbeddings(
    model_name="flax-sentence-embeddings/all_datasets_v3_mpnet-base")
faiss_db = FAISS.load_local("faiss.db", embeddings)

chain = get_chain(faiss_db)
result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
