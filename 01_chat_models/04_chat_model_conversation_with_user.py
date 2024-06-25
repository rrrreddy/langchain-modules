from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# initialize the model

model = ChatAnthropic(model="claude-3-opus-20240229")

# create a list to store chat histrory

chat_history = []

# initialize a system message.

system_message = SystemMessage(content="You are an helpfull assistant, Please asnwer to the questions.")
chat_history.append(system_message)


while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    
    # get response based on chat history 
    result = model.invoke(chat_history)
    response = result.content
    # add the ai respone to chat_chistory
    chat_history.append(AIMessage(content=response))
    
    print(f"AI respone: {response}")
    
    

print("---------Message History / Chat Conversation-------------")
print(chat_history)
    
    
