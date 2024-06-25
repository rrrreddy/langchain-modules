from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# PART 1: Create a PromptTemplate using a template string

template = "Tell me a joke about {topic}."

prompt_template = PromptTemplate.from_template(template=template)

prompt1 = prompt_template.invoke({"topic": "Python"})

print(prompt1)


# PART 2: Prompt with Multiple Placeholders
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:"""
prompt_multiple = PromptTemplate.from_template(template_multiple)
prompt2 = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
print("\n----- Prompt with Multiple Placeholders -----\n")
print(prompt2)


# PART 3: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt3 = prompt_template.invoke({"topic": "lawyers", "joke_count": 10})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt3)


# Extra Informoation about Part 3.
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 5 jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)
