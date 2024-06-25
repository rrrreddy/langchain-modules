from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.schema.runnable import RunnableLambda, RunnableSequence


# model 


model = ChatAnthropic(model="claude-3-opus-20240229")


# prompt

template = (
    [
    ("system", "You are a funny scientist who funny facts about {topic}."),
    ("human", "tell me {funny_facts_count} funny facts"),
    ]
)

prompt_template = ChatPromptTemplate.from_messages(messages=template)


# create custom runnables for the chains

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))

parse_output  = RunnableLambda(lambda x: x.content)


# create a runnable sequence chain 

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke(
    {
            "topic": "robots",
        "funny_facts_count": 3
    }
)

print(response)


