from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

# Load environment variables from .env

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Create a Groq model Model:llama3-8b-8192

model = ChatGroq(model="llama3-8b-8192")

# now lets invoke the model with a message 

result = model.invoke("what is the summation of the numbers: 123, 10, 10, 100")

print("Full Result:")
print(result)
print("Content Only:")
print(result.content)

