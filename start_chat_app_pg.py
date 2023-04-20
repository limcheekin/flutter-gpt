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
import os
from langchain.vectorstores.pgvector import PGVector

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.channels import Channels
from models.conversations import Conversations
from models.messages import Messages
from models.models import Models

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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


user = {
    "id": 1,
    "channel": "android",
    "message": "No message defined"
}

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("DB_DRIVER", "psycopg2"),
    host=os.environ.get("DB_HOST", "localhost"),
    port=int(os.environ.get("DB_PORT", "5432")),
    database=os.environ.get("DB_NAME", "postgres"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", "postgres"),
)

# REF: https://www.compose.com/articles/using-postgresql-through-sqlalchemy/
db = create_engine(CONNECTION_STRING)
Session = sessionmaker(db)
session = Session()

# Remove the comment below for table creations
# models.DeclarativeBase.metadata.create_all(db)

channels = session.query(Channels).all()
channels_name_id = {c.name: c.id for c in channels}

models = session.query(Models).all()
models_dict = {m.name: Models.to_dict(m) for m in models}


def find_or_create_conversation(user):
    channel_id = channels_name_id[user['channel']]
    conversation = session.query(Conversations).filter_by(
        user_id=user['id'], channel_id=channel_id).first()
    if conversation is None:
        conversation = Conversations(
            user_id=user['id'], channel_id=channel_id,
            context=f"User {user['id']} from {user['channel']}")
        session.add(conversation)
        session.commit()
    return conversation


def create_message(user, conversation, result):
    ids = [doc.metadata['id'] for doc in result['source_documents']]
    sources = [doc.metadata['source'] for doc in result['source_documents']]
    message = Messages(
        conversation_id=conversation.id,
        user_id=user['id'],
        user_message=user['message'],
        bot_message=result['answer'],
        model_id=models_dict[MODEL_NAME]['id'],
        cmetadata={"id": ids, "source": sources}
    )
    session.add(message)
    session.commit()
    return message


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


def get_chain(memory):
    llm = get_llm()
    doc_chain = load_qa_with_sources_chain(
        llm, chain_type="stuff", verbose=True)
    chain = ConversationalRetrievalChain(
        question_generator=get_question_generator(llm),
        combine_docs_chain=doc_chain,
        retriever=get_retriever(k=1),
        memory=memory,
        get_chat_history=get_chat_history,
        max_tokens_limit=models_dict[MODEL_NAME]['max_tokens'],
        return_source_documents=True)
    return chain


def get_memory(conversation):
    messages = session.query(Messages).filter_by(conversation_id=conversation.id).order_by(
        Messages.created_at.desc()).limit(os.environ.get("CONVERSATIONAL_MEMORY_WINDOW_SIZE", 3)).all()
    messages.reverse()
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer")
    for message in messages:
        memory.chat_memory.add_user_message(message.user_message)
        memory.chat_memory.add_ai_message(message.bot_message)
    return memory


if __name__ == "__main__":
    conversation = find_or_create_conversation(user)
    qa_chain = get_chain(get_memory(conversation))
    print("Chat with the FlutterGPT bot:")
    print(f"Conversation: {vars(conversation)}")
    while True:
        print("Your question:")
        user['message'] = input()
        result = qa_chain({"question": user['message']})
        print(f"Answer: {result['answer']}")
        print(f"source_documents: {result['source_documents']}")
        message = create_message(user, conversation, result)
        print(vars(message))
        print("-" * 80)
