# from langchain_openai import OpenAI
# from dotenv import load_dotenv
# import os
# load_dotenv()

# llm1 = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"))

# response = llm1.invoke("hi")
# print(response)
# from langchain_openai import OpenAI
# from dotenv import load_dotenv
# import os
# load_dotenv()

# llm1 = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"))

# response = llm1.invoke("hi")
# print(response) index with llm
# while True:
#     human_message = input("Ask any question about hotel")
#     response = index.query(human_message, llm=llm)
#     print(response) 

#memory

# from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory
# from langchain.chains import ConversationChain
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os

# load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# memory = ConversationBufferMemory()
# memory1 = ConversationBufferWindowMemory(k=5)
# memory2 = ConversationSummaryMemory(llm=llm)
# memory3 = ConversationSummaryBufferMemory(llm=llm,max_token_limit=100)




# chain = ConversationChain(memory=memory3,llm=llm)

# while True:
#     user_input = input("You:-")
#     if user_input == "exit":
#         break
#     response = chain.invoke(user_input) 
#     print("Final ==>>", response)   

# Rag

# from langchain_community.document_loaders import TextLoader
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# import os
# from dotenv import load_dotenv
# load_dotenv()

# llm = GoogleGenerativeAI(
#     model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
# try:
#     loader =  TextLoader("data.txt")  
# except Exception as e:
#     print("Error while loading file=", e)   

# # Create embeddings
# embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # Use a smaller chunk size to manage token limit
# text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=200)

# # Create the index with specified model and text splitter
# index_creator = VectorstoreIndexCreator(
#     text_splitter=text_splitter,
#     embedding=embedding
# )

# index = index_creator.from_loaders([loader])

# # Query the index with llm
# response = index.query("What is the name of your school?", llm=llm)
# print(response)

from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Load data.txt
file_path = "langchain/data.txt"

try:
    loader = TextLoader(file_path, encoding="utf-8")  
    print("File loaded successfully!")
except Exception as e:
    print("Error while loading file:", e)
    exit()  # Stop execution if file fails to load


# Create embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Use a smaller chunk size
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=200)

# Create index
index_creator = VectorstoreIndexCreator(
    text_splitter=text_splitter,
    embedding=embedding
)

# Ensure 'loader' exists before using it
index = index_creator.from_loaders([loader])

# Query the index with LLM
while True:
    human_message = input("Ask any question about the school: ")
    response = index.query(human_message, llm=llm)
    print(response)
