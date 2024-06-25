from dotenv import load_dotenv
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

# create chat model 

model = ChatGroq(model="llama3-8b-8192")

# SystemMessage:
# Message for AI model to behave as per the instructions, usually passed as the first of the sequence of the input messages.

# HumanMessage:
# Message from a human to the AI model , just like an question to the model
# AIMessage : Response/message from the AI model based on the SystemMessage and HumanMessage

messages = [
    SystemMessage(content="You are a math problem solve, solve the following math problem:"),
    HumanMessage(content="what is the product of 100 and 250?"),
]

# Invoke the model 

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")


# let's update messages and see 
messages = [
     SystemMessage(content="You are a math problem solve, solve the following math problem:"),
     HumanMessage(content="what is the product of 100 and 250?"),
     AIMessage(content="Easy one!The product of 100 and 250 is:100 x 250 = 25,000"),
     HumanMessage(content="what is 100 divided by 3.5?"),
]


result = model.invoke(messages)

print(f"Answer from AI: {result.content}")

