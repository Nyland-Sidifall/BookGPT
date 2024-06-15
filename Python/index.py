import os
from dotenv import load_dotenv
import getpass
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Question to ask
Question = input("What's your question for GPT? ")

#load env variables
load_dotenv() 

#Load up PDF's into doc loaded
loader = PyPDFDirectoryLoader("Books/")
docs = loader.load()


# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY") 

# Verify if the key is loaded (Optional)
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

#set up open AI connection
#os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key: ")

#init LLM
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0,api_key=openai_api_key)

#Split pdf's
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()


# Define system template
system_template = (
    "You are an assistant for question-answering tasks. "
    "You are also an expert programmer who specializes in Python Programming. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know.\n\n"
    "{context}"
)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{input}"),
])

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": Question})

print(results["answer"])