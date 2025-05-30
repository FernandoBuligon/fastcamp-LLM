from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

# Tools
youtube_tool = YouTubeSearchTool()
google_trends = GoogleTrendsQueryRun(api_wrapper=GoogleTrendsAPIWrapper())
tools = [youtube_tool, google_trends]

# Prompt
prompt = hub.pull('hwchase17/openai-functions-agent')

# LLM (usando Ollama)
llm = ChatOllama(model="mistral")

# Corrigido: usando create_tool_calling_agent
agente = create_tool_calling_agent(llm, tools, prompt=prompt)

# Executor
agent_executor = AgentExecutor(agent=agente, tools=tools, verbose=True)

agent_executor.invoke({'input': 'youtube_search[Quero que ache links das aulas de algoritmos e estruturas de dados do thiago naves]'})
