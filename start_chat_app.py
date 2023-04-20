import pickle
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, set_seed
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
load_dotenv()

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
system_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.
Use the standalone question to answer the user's question. 
If you don't know the answer, just say "Hmm..., I'm not sure.", don't try to make up an answer.
Conversation:
{chat_history}
Follow up question: {question}
Standalone question:"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)


def get_question_generator(llm):
    question_generator = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return question_generator


def get_llm(
    min_length: int = 20,
    max_length: int = 200,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50
):
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        min_length=min_length,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm


def get_retriever(k: int = 3):
    with open("faiss_store.pkl", "rb") as f:
        vector_store = pickle.load(f)

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever


def get_chat_history(history) -> str:
    print(f"CHAT HISTORY:\n{history}")
    return history


def get_chain():
    llm = get_llm()
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer")
    doc_chain = load_qa_with_sources_chain(
        llm, chain_type="stuff", verbose=True)
    chain = ConversationalRetrievalChain(
        question_generator=get_question_generator(llm),
        combine_docs_chain=doc_chain,
        retriever=get_retriever(k=1),
        memory=memory,
        get_chat_history=get_chat_history,
        max_tokens_limit=250,
        return_source_documents=True)
    return chain


if __name__ == "__main__":
    qa_chain = get_chain()
    print("Chat with the FlutterGPT bot:")
    while True:
        print("Your question:")
        question = input()
        result = qa_chain({"question": question})
        print(f"Answer: {result['answer']}")
        print(f"\nResult: {result}")
        print("-" * 80)
