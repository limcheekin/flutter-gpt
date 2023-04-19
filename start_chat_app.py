import pickle
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, set_seed
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


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
    memory = ConversationBufferMemory(memory_key="chat_history")
    chain = ConversationalRetrievalChain.from_llm(
        get_llm(), get_retriever(k=1), memory=memory,
        get_chat_history=get_chat_history, verbose=True)
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
