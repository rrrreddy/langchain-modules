from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from langchain.schema.runnable import RunnableLambda


# model 


model = ChatAnthropic(model="claude-3-opus-20240229")


# prompt

template = (
    [
    ("system", "You are a funny scientist who funny facts about {topic}."),
    ("human", "tell me {funny_facts_count} funny facts"),
    ]
)

prompt = ChatPromptTemplate.from_messages(messages=template)


output_parser = StrOutputParser()


uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"word count: {len(x.split())}\n {x}")

# create chain


chain = prompt | model | output_parser | uppercase_output | count_words

result = chain.invoke(
    {
        "topic": "robots",
        "funny_facts_count": 3
    }
)


print(result)