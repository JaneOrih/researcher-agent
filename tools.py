#from langchain_core.prompt import Prompt
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


search= DuckDuckGoSearchRun()
search_tool= Tool(
    name = 'search', #no space  
    func= search.run,
    description = "search the web for information"
)
