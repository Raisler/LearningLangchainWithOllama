from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import create_async_playwright_browser
from langchain.llms import Ollama
from langchain.agents import AgentExecutor, AgentType

# Create the PlayWright browser toolkit
toolkit = PlayWrightBrowserToolkit(async_browser=create_async_playwright_browser())

# Instantiate the Ollama language model
llm = Ollama(model="llama2:13b")

# Create the agent executor
agent_executor = AgentExecutor(
    agent_type=AgentType.PLAYWRIGHT,
    toolkit=toolkit,
    llm=ollama,
    verbose=False
)
# Run the agent with your desired prompts
prompts = [
    'Make a screenshot of the page and save in the directory ".", The website I want the screenshot is this: langchain.com'
]

for prompt in prompts:
    response = agent.run(prompt)
    print(response)