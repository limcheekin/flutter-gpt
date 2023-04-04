import pickle
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain, LLMChain

system_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.
Use the standalone question to answer the user's question. 
If you don't know the answer, just say "Hmm..., I'm not sure.", don't try to make up an answer.
Don't return a "SOURCES" part in your answer.

Chat history:
{chat_history}

Follow up question: {question}

Standalone question:"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


def get_chain(store):
    llm = ChatOpenAI(temperature=0)
    question_generator = LLMChain(llm=llm, prompt=prompt)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

    chain = ConversationalRetrievalChain(
        retriever=store.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain
    )
    return chain


if __name__ == "__main__":
    with open("faiss_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with the FlutterGPT bot:")
    while True:
        print("Your question:")
        question = input()
        result = qa_chain({"question": question, "chat_history": chat_history})
        print(f"Answer: {result['answer']}")
        chat_history.append((question, result["answer"]))
        print(f"Result: {result}")
