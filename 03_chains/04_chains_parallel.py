from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel


load_dotenv()

model = ChatAnthropic(model='claude-3-opus-20240229')

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)



# create a chain which can parallely checks both pros and cons for a product 
# main prompt template for the prodct 
# pros prompt template
# cons prompt template
# https://python.langchain.com/v0.2/docs/how_to/parallel/



# Define pros analysis step

def analyze_pros(features):
    pros_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are an expert product reviewer."),
            ("human", "Given these features: {features}, list the pros of these features."),
        ]
    )
    
    return pros_prompt_template.format_prompt(features=features)

# Define cons analysis step

def analyze_cons(features):
    cons_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are an expert product reviewer."),
            ("human", "Given these features: {features}, list the cons of these features."),
        ]
    )
    
    return cons_prompt_template.format_prompt(features=features)


def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n Cons:\n{cons}"


pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)


# Combined chain using LCEL 

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros":pros_branch_chain, "cons":cons_branch_chain})
    |RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# run the chain

result = chain.invoke(
    {
        "product_name" : "Apple 14 pro max"
    }
)


print(result)
    