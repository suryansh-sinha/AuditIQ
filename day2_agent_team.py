import os
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Defining the model client
model_client = OpenAIChatCompletionClient(
    model='gemini-2.5-flash',
    api_key=api_key
)

# Research Assistant
research_assistant = AssistantAgent(
    name="research_assistant",
    model_client=model_client,
    system_message="You are a helpful AI research assistant for a tech startup. You bridge the gap between academic research and product innovation."
)

# Problem Analyst
problem_analyst = AssistantAgent(
    name="problem_analyst",
    model_client=model_client,
    system_message="You analyze business problems and break them down into searchable research keywords."
)

# Keyword Specialist
keyword_specialist = AssistantAgent(
    name="keyword_specialist",
    model_client=model_client,
    system_message="You refine keywords to be optimal for academic database searches like ArXiv."
)

# termination_condition = TextMentionTermination("APPROVED")

# Research Team
research_team = RoundRobinGroupChat(
    participants=[problem_analyst, keyword_specialist],
    max_turns=2
)

# Test the agent
async def analyze_problem():
    task = "I want to solve 'real-time fraudulent transaction detection'. Break this down and suggest 5 good keywords for an ArXiv search."
    await Console(research_team.run_stream(task=task))
    await model_client.close()

asyncio.run(analyze_problem())