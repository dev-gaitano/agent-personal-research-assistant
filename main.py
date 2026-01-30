from dotenv import load_dotenv
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool, tool
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
import wikipedia

load_dotenv()

# Model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=1.0,
    max_retries=2
)

# Tools
@tool("search_engine", description="use to find related information on a topic", return_direct=False)
def search_engine(research_topic: str) -> list:
    try:
        search = DuckDuckGoSearchResults(output_format="list")
        search_results: list[dict] = search.invoke(research_topic)
        return search_results
    except Exception as e:
        print(f"Search Engine Error: {e}")
        return [{ "Search Engine Error" : e }]

@tool("wikipedia", description="Use to research details on a topic", return_direct=False)
def wikipedia(research_topic: str) -> str:
    try:
        wiki = WikipediaAPIWrapper(wiki_client=wikipedia)
        wikipedia_result = wiki.run(research_topic)
        return wikipedia_result
    except Exception as e:
        print(f"Wikipedia Error: {e}")
        return f"Wikipedia Error: {e}"

tools: list[BaseTool] = [ search_engine, wikipedia ]

# Agent
SYSTEM_PROMPT: str = """
    You are an extremely intellligent and fact based research assistant.
    You help users understand topics through thorough research and respond ith
    clear structured summaries.

    You have access to a few tools that you can leverage to get the most optimal
    results:

    - search_engine : Use this to search and crawl the web to get insights on a
      said topic.
    - wikipedia -  Use this to get more factual info on the topic at hand 
""" # TODO: Set a min and max word limit, save output to a file, add citations

@dataclass
class ResponseFormat:
    content: str

agent = create_agent(
    model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=ResponseFormat
)

try:
    raw_response: dict = agent.invoke({
        "messages": [
            {
            "role": "user",
            "content": "Approaches to developing a version control system"
            },
        ]
    })

    print(raw_response["structured_response"].content)
except Exception as e:
    print(f"Invocation Error: {e}")
