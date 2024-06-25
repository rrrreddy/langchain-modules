import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline



from langchain_core.messages import AIMessage, SystemMessage, HumanMessage



# find the list of models that works with langchain 
# https://python.langchain.com/v0.2/docs/integrations/chat/


# load env and setup messages 

load_dotenv()

messages = [
    SystemMessage(content="You are a math problem solver. only give the direct answer to the question, solve the following math problem:"),
    HumanMessage(content="what is the sum of 20 and 40?"),
]

# ---- Anthropic Chat Model  ----
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview

model = ChatAnthropic(model="claude-3-opus-20240229")


result = model.invoke(messages)

print(f"Anthropic model result: {result.content}")


# ---- Cohere Chat Model  ----
# https://dashboard.cohere.com/

model = ChatCohere(model="command-r-plus")

result = model.invoke(messages)

print(f"Cohere model result: {result.content}")

# ---- Groq Chat Model  ----
# https://console.groq.com/docs/quickstart

model = ChatGroq(model="llama3-70b-8192")

model.invoke(messages)

print(f"Groq llama3 model result: {result.content}")



# ------- HuggingFace mistralai/Mistral-7B-Instruct-v0.3 model -------
#   https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

# hugging face we can instantiate in different ways

# 1. Instantiate an LLM 

# using HuggingfaceEndpoint

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    repetition_penalty=1.03,
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# result = llm1.invoke(messages)
# print(f"HuggingFaceEndpoint mistral-7B model result: {result}")



# using HuggingFacePipeline

# model_id = "EleutherAI/gpt-neo-2.7B"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)

# llm2 = HuggingFacePipeline(pipeline=pipe)

# result = llm2.invoke(messages)

# print(f"HuggingFacePipeline gpt2 model result: {result} ")




# ChatHuggingFace  model

# we can use any llm here huggingfacepipeline llm or huggingfaceendpoint llm in the chatmodel
# here I am using llm1 i.e mistralai/Mistral-7B-Instruct-v0.3 


model = ChatHuggingFace(llm=llm1)
result = model.invoke(messages)

print(f"ChatHuggingFace result: {result.content}")
