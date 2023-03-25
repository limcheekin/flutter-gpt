import pickle
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

system_template="""Use the following pieces of context to answer the users question. 
If you don't know the answer, just say "Hmm..., I'm not sure.", don't try to make up an answer.
ALWAYS return a "Sources" part in your answer.
The "Sources" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo

Sources: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

def get_chain(vectorstore, prompt):
    chain_type_kwargs = {"prompt": prompt}
    chain = VectorDBQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
        chain_type="stuff", 
        vectorstore=vectorstore,
        chain_type_kwargs=chain_type_kwargs
    )
    return chain


if __name__ == "__main__":
    with open("faiss_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_chain(vectorstore, prompt)
    print("Chat with the FlutterGPT bot:")
    while True:
        print("Your question:")
        question = input()
        result = qa_chain({"question": question}, return_only_outputs=True)
        print(f"Answer: {result['answer']}")
        print("\n\n")
        print(result)