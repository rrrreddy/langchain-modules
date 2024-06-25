from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

def get_current_time(*args, **kwargs):
    """Get the current time in H:MM AM/PM format."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


# tools list avaibale to the agent

tools =[
    Tool(
        name="Time", # name of the tool
        func=get_current_time, # function that the tool will execute
        # descrption of the tool
        description="Useful for when you need to know the current time",
    ),
]


# pull the prompt from the hub
# https://smith.langchain.com/hub/hwchase17/react?organizationId=267264aa-7dba-58ef-b682-fb6886f1c02e
#ReAct = Reason and Action 

prompt = hub.pull("hwchase17/react")


# llm model 

llm = ChatAnthropic(
    model = "claude-3-opus-20240229",
    temperature=0
)

# create ReAct agent using the create_react_agent function

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# create an agent executor fro mthe agent and tools 

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

response = agent_executor.invoke(
    {
        "input": "what time is it?"
    }
)

print(response)