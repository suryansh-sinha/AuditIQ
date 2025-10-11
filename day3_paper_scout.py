import os
import arxiv
import asyncio
from dotenv import load_dotenv
from arxiv import SortCriterion
from typing import Annotated, List

from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

model_client = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    api_key=api_key
)

async def search_arxiv(
    query: Annotated[str, "Search query for ArXiv papers"],
    max_results: Annotated[int, "Maximum number of results"] = 5
) -> str:
    """Search arxiv for academic papers and return formatted results."""
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=SortCriterion.Relevance,
    )

    results: List[str] = []

    for paper in client.results(search):
        author_names = [author.name for author in paper.authors]
        author_display = f"{author_names[0]} et al." if len(author_names) > 1 else author_names[0]
        result_string = (
            f"**Title:** {paper.title}\n"
            f"**Authors:** {author_display}\n"
            f"**URL:** {paper.pdf_url}\n"
            f"**Abstract:** {paper.summary[:500]}...\n"
        )
        results.append(result_string)

    out = "\n\n" + ("\n\n" + "="*50 + "\n\n").join(results) if results else "No papers found."
    return out

# Register the tool
arxiv_tool = FunctionTool(
    func=search_arxiv,
    description="A tool to fetch top 5 research papers from ArXiv based on the given query, sorted by relevance.",
    name="arxiv_tool"
)

# Create the Paper Scout agent with the tool
paper_scout = AssistantAgent(
    name="paper_scout",
    model_client=model_client,
    tools=[arxiv_tool],
    system_message="You are a research specialist. Use the ArXiv search tool to find relevant papers based on keywords."
)

# Testing the agent
async def agent_test():
    task = "Find the top 2 papers on 'graph neural networks fraud detection'"
    await Console(paper_scout.run_stream(task=task))
    await model_client.close()

# Run the test function
asyncio.run(agent_test())