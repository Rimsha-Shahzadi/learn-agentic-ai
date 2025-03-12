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

# from langchain_community.document_loaders import TextLoader
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # Initialize LLM
# llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# # Load data.txt
# file_path = "langchain/data.txt"

# try:
#     loader = TextLoader(file_path, encoding="utf-8")  
#     print("File loaded successfully!")
# except Exception as e:
#     print("Error while loading file:", e)
#     exit()  # Stop execution if file fails to load


# # Create embeddings
# embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # Use a smaller chunk size
# text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=200)

# # Create index
# index_creator = VectorstoreIndexCreator(
#     text_splitter=text_splitter,
#     embedding=embedding
# )

# # Ensure 'loader' exists before using it
# index = index_creator.from_loaders([loader])

# # Query the index with LLM
# while True:
#     human_message = input("Ask any question about the school: ")
#     response = index.query(human_message, llm=llm)
#     print(response)

#Lecture#12
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain_core.tools import tool
# import requests
# from langchain_core.runnables import RunnableSequence

# llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyBNECCwmlO5GbGgrrBg3NPimvrJroUVanA")
# result = llm.invoke("What is the capital of France?")
# finalOutput = llm.invoke(f"what is teh temperature of city, {result}")

# prompt_template = PromptTemplate(
#     input_variables=["input"],
#     template="you ave a tool caller, you have to call the tool named add_numbers_tool if there is any addition required, please don't send any explanation while calling the function, just send two numbers what users provided e.g 50,50, even though user give sentence you you have to find two numbers and pass to the numbers user input is: {input}/n")

# @tool
# def get_weather_tool(city: str) -> str:
#     """Get the weather of city  """
#     print("get_weather_tool_input_data", city)
#     output = request(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_Key}")
#     return output
# @tool
# def add_numbers_tool(input_data: str) -> str:
#     """ addition of two numbers. """
#     print("Add_numbers_tool_input_data", input_data)
#     return "your result is 100"
#     try:
#         numbers = input_data.split(',')
#     except Exception as e:
#         return "No numbers found"
#     num1, num2 = int(numbers[0], int(numbers[1]))  
#     result = num1 + num2
#     return f"The sum of {num1} and {num2} is {result}"

# @tool
# def add_multiply_tool(input_data: str) -> str:
#     """ Multipy of two numbers """
#     print("multiply_tool_input_data", input_data)
#     return "multiplication is  200"
        

# chain = RunnableSequence(
#     prompt_template,
#     llm,
#     add_numbers_tool,
#     add_multiply_tool
# )
# output = chain.invoke("my first value is 50 and second is minus 5")
# print("output", output)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.human import HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


def deposit_money(name:str , bank_account:str, amount:int) -> dict:
    """Deposit Money in Bank Account
    Args:
        name (str): Name of the person
        bank_account (str): Bank Account Number
        amount (int): Amount of money to be deposited
    Returns:
        dict: a dict
    """
    return {
        "status":f"Successfully Deposited {amount} in {bank_account} for {name}"}

toolwalallm = llm.bind_tools([deposit_money])

finalOutput = toolwalallm.invoke([HumanMessage(content=f"Deposit 1000 in 1234567890 for John")])

print(finalOutput)