import os

# import to load environment variables
from dotenv import load_dotenv

# import for prompt template and message

from langchain import hub
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


# import for chat model
from langchain_cohere import ChatCohere

# import for document loaders

from langchain.document_loaders import WebBaseLoader

# import for text splitter

from langchain.text_splitter import RecursiveCharacterTextSplitter

# import for embedding

# import for Vector Store
from langchain.vectorstores import Chroma

# import chain chains and reterivers
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever


from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

from langchain_core.messages import AIMessage,HumanMessage



load_dotenv()

# initialize the model

model = ChatCohere(model="command-r", temperature=0.3)

# load vector store

from langchain_pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

index_name = "got-pinecone-index"

# initialize the embedding model
    
from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'mps'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    show_progress=True,
    )

# initialize the vector store

vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding_model)

# query = "Who is the father of Arya Stark?"

# initialize the retriever


search_type = "similarity"
retriever =vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5},
)


# contextualizing the question prompt

contextualiz_system_question = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualiz_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualiz_system_question),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


# Create a history-aware retriever

retriever = create_history_aware_retriever(
    llm=model,
    retriever=retriever,
    prompt=contextualiz_question_prompt,
)


# Answer question prompt

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm=model, prompt=qa_prompt)

rag_chain = create_retrieval_chain(
    retriever=retriever, combine_docs_chain=question_answer_chain
)

# Set Up ReAct Agent with Document Store Retriever
# Load the ReAct Docstore Prompt
react_docstore_prompt = hub.pull("hwchase17/react")


tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="useful for when you need to answer questions about the context",
    )
]

# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history})
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))