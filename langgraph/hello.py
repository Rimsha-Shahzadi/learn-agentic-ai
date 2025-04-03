# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages.human import HumanMessage
# from dotenv import load_dotenv
# import os
# load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


# def deposit_money(name:str , bank_account:str, amount:int) -> dict:
#     """Deposit Money in Bank Account
#     Args:
#         name (str): Name of the person
#         bank_account (str): Bank Account Number
#         amount (int): Amount of money to be deposited
#     Returns:
#         dict: a dict
#     """
#     return {
#         "status":f"Successfully Deposited {amount} in {bank_account} for {name}"}

# toolwalallm = llm.bind_tools([deposit_money])

# finalOutput = toolwalallm.invoke([HumanMessage(content=f"hi")])

# print(finalOutput)




#langgraph
# from typing_extensions import TypedDict
# class LearningState(TypedDict):
    # prompt: str

# Lahore_state = LearningState(prompt="Hello Rimi From Lahore")   

# print(Lahore_state)
# print(Lahore_state['prompt'])
# print(Lahore_state['prompt'] + "How are you?")

# def node_1(state: LearningState) -> LearningState:
#     print("---Node 1 state---", state)
#     return {"prompt": state["prompt"] + "I am"}
# def node_2(state: LearningState) -> LearningState:
#     print("---Node 2 state---", state)
#     return {"prompt": state["prompt"] + "fine"}

from IPython.display import Image, display 
from langgraph.graph import StateGraph, START , END
from langgraph.graph.state import CompiledStateGraph 

builder:StateGraph = StateGraph(state_schema=LearningState)
print(type(builder))





   




