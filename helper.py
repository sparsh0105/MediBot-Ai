# Download all the necessary libraries
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings

from supabase import create_client
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from datetime import datetime
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer

# load env
load_dotenv()

# Setup supabase (for storing chat history)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class SupaBaseChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self):
        # retrieve all messages from supabase
        res = supabase.table('medical-convo').select("*").eq("session_id", self.session_id).order("timestamp").execute()
        return [HumanMessage(content=msg['content']) if msg['role'] == 'user'
                else AIMessage(content=msg['content'])
                for msg in res.data]

    def add_message(self, message: BaseMessage):
        # Required by LangChain - handles both user and AI messages
        if isinstance(message, HumanMessage):
            self._store_message("user", message.content)
        elif isinstance(message, AIMessage):
            self._store_message("ai", message.content)
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
        
    # Aliases for backward compatibility
    add_user_message = lambda self, message: self.add_message(HumanMessage(content=message))
    add_ai_message = lambda self, message: self.add_message(AIMessage(content=message))

    # For storing the message in database
    def _store_message(self, role: str, content: str):
        supabase.table("medical-convo").insert({
            "id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }).execute()

    def clear(self):
        supabase.table("medical-convo").delete().eq("session_id", self.session_id).execute()

# QDRANT(stores all our medical documents) + Retriever (making the database as retriever to get relevant documents) 
def get_vectorstore():
    client = QdrantClient(
        os.getenv('QDRANT_HOST'),
        api_key=os.getenv('QDRANT_API_KEY')
    )

    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    
    vector_store = Qdrant(
        client=client,
        collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
        embeddings=embeddings
    )

    return vector_store


# Create the final chain
def get_conversational_chain():
    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(search_kwargs={'k': 15})

    # Configure Gemini LLM
    from google import generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # History-aware retriever
    retriever_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a medical query rewriter. "
            "Your job is to transform the user's latest message into a complete, clear medical search query "
            "by combining all relevant details from the chat history. "
            "Include all symptoms, affected body parts, possible conditions, duration, and severity if mentioned. "
            "Be explicit — for example: "
            "'User: I have red spots.' -> Rewrite: 'Patient has itchy red spots on trunk and scalp, possible viral rash.' "
            "Use medical terms if clear. "
            "If the user message is vague, use chat history to add context. "
            "IMPORTANT: Respond ONLY with the rewritten query. Do not add explanations, notes, or extra text."
            ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
        ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, retriever_prompt)

    # System prompt for final answer
    system_prompt = (
        "You are a trusted medical assistant. "
        "Your task is to answer the user's question using only the provided context below. "
        "Always explain your reasoning clearly and professionally, using simple medical language if needed. "
        "If the context does not contain enough information to answer reliably, reply: 'I do not know based on the provided information.' "
        "Do not make up any information or medical advice beyond what is given in the context. "
        "Keep your answers clear, precise, and concise.\n\n"
        "Context:\n{context}"
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="chat_history")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, chat_prompt)

    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    def format_output(response):
        answer = response["answer"].replace("- [ ]", "— ◻").replace("- [x]", "— ✅")
        return {
            "output": answer,
            "context": response["context"],
            "input": response["input"]
        }

    formatted_chain = retrieval_chain | format_output

    conversational_chain = RunnableWithMessageHistory(
        formatted_chain,
        lambda session_id: SupaBaseChatHistory(session_id),
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='output'
    )

    return conversational_chain

