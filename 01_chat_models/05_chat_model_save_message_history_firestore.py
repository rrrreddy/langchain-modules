from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory




# langchain firestore instructions : https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore/
# create a google cloud project : https://developers.google.com/workspace/guides/create-project
# enable firestore api key : https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=langchain-chatbot-427503
# create fire store databse : https://cloud.google.com/firestore/docs/manage-databases

# firestore details 

#project_name = "langchain-chatbot"
#project_id = "langchain-chatbot-427503"

# steps to follow after project creation
# install google cli - https://cloud.google.com/sdk/docs/install
# unzip and then run below commands
# ./google-cloud-sdk/install.sh
# ./google-cloud-sdk/bin/gcloud init
# now select the project you want to use fromo the list in my case it is langchain-chatbot-427503


# path variable setup 
# export PATH="$PATH:/Users/[path]/google-cloud-sdk/bin"
# source ~/.zshrc

load_dotenv()


#setup firestore

PROJECT_ID = "langchain-chatbot-427503"
SESSION_ID = "Raghu-Session-01"
COLLECTION_NAME = "chat_history"


# initialize firestore
print("Initializing Firestore Client...")

client = firestore.Client(project=PROJECT_ID)


# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")

chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
    encode_message=True
)

print("---------Chat History is Initialized---------")
print(f"current chat history:{chat_history.messages}")

# initialize chatmodel

model = ChatAnthropic(model="claude-3-opus-20240229")

print("Start chatting with the AI. Type 'exit' to quit.")


while True:
    query = input("User: ")
    if query.lower()=="exit":
        break
    
    chat_history.add_user_message(query)
    
    # if len(chat_history.messages)>10:
    #     respone = model.invoke(chat_history.messages[-10:])
    # else:
    respone = model.invoke(chat_history.messages)

    chat_history.add_ai_message(respone.content)
    print(f"AI Respone: {respone.content}")
    



