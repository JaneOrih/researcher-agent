import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

#from tools import search_tool

from datetime import datetime

load_dotenv()

# Initialize the chat model with the API key
grok_api_key = os.getenv("GROQ_API_KEY") 
if grok_api_key is None:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")
llm = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=grok_api_key)

#response = llm.invoke("whats the generic pattern to make Nigerian food")

class researchResponse(BaseModel):
    response: str 
    summary: str
    sources:list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=researchResponse)


SYSTEM_PROMPT= """
you are a researcha assistant that will help generate research papers
Answer the user query and ause the necessary tools
wrap the output in this format and provide no other text \n {format_instructions}
"""
prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}")
            #MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(format_instructions= parser.get_format_instructions())



agent= create_tool_calling_agent(
    llm=llm,
    prompt= prompt,
    tools= []
)

agent_executor= AgentExecutor(agent= agent, tools=[], verbose= True)
raw_response= agent_executor.invoke({"query":"whats the generic pattern to make Nigerian food"})
print(raw_response)

try:
    structured_response= parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("error parsing response", e, "Raw response: ", raw_response)