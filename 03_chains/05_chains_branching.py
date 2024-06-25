from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch


load_dotenv()

model = ChatAnthropic(model='claude-3-opus-20240229')

#define prompt template for different feedbacks

# positive feedback template

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "Generate a psotive note for this positive feedback: {feedback}."),
    ]
)


# Negative feedback template

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "generate a response addressing this negative feedback: {feedback}."),
    ]
)


# Neutral feedback template

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a request for more detail for this neutral feedback: {feedback}."),
    ]
)

# escalate feedback template

escalte_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a message to escale this feedback to human agent: {feedback}."),
    ]
)



# feedback classification template


classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or esclated: {feedback}."),
    ]
)


feedback_classification_chain = classification_template | model | StrOutputParser()



# create brances to pass the classified feedback to specific branch

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()  # +ve feedbcka chain
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser() # negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser() # Neutral feedback chain
    ),
    escalte_feedback_template | model | StrOutputParser()   # if this feedback is not any of the above then by default it goes to escalted  
)

chain = feedback_classification_chain | branches


# Run the chain with an example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "The product is terrible. It broke after just one use and the quality is very poor."

result = chain.invoke({"feedback": review})

print(result)
