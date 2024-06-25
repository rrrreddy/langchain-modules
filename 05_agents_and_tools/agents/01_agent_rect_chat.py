from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_anthropic import ChatAnthropic


# Load environment variables
load_dotenv()

def get_current_time(*args, **kwargs):
    """
    Get the current time in H:MM AM/PM format.

    This function imports the datetime module and uses it to get the current time.
    It then formats the current time as H:MM AM/PM and returns it as a string.

    Parameters:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        str: The current time in H:MM AM/PM format.
    """
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    """
    A function that searches Wikipedia for a summary based on the provided query.

    Parameters:
        query (str): The search query to look up on Wikipedia.

    Returns:
        str: A summary of the Wikipedia page related to the query, limited to 2 sentences. If no information is found, a default message is returned.
    """
    from wikipedia import summary
    try:
        return summary(query,sentences=2)
    except:
        return "I coundn't find any information on that."


tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="userful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="useful for when you need to know information about the topic.",
    ),
]


prompt =hub.pull("hwchase17/structured-chat-agent")

llm = ChatAnthropic(model = "claude-3-opus-20240229")


# memory = ConversationBufferMemory(
#     memory_key="chat_history", return_messages=True
# )


agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    # memory=memory,
    handle_parsing_errors=True, 
)



# initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."

# memory.chat_memory.add_message(SystemMessage(content=initial_message))


#chat in loop until exit

while True:
    user_input = input("User: ")
    
    if user_input.lower()=="exit":
        break
    
    # memory.chat_memory.add_message(HumanMessage(content=user_input))
    # print("Messages before invoking:", [msg.content for msg in memory.chat_memory.messages])

    response = agent_executor.invoke({"input": user_input})
    # print("Messages after invoking:", [msg.content for msg in memory.chat_memory.messages])

    print("Bot:", response["output"])
    # memory.chat_memory.add_message(AIMessage(content=response["output"]))