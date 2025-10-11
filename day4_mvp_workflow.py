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
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

model_client = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    api_key=api_key
)

async def search_arxiv(
    query: Annotated[str, "Search query for ArXiv papers"],
    max_results: Annotated[int, "Maximum number of results"] = 10
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
    description="A specialized tool to fetch the **top 10 research papers** from ArXiv based on the given search query, with results sorted by relevance. The output is a raw string containing the title, authors, URL, and a summary for each paper.",
    name="arxiv_tool"
)


# Create agents 
problem_analyst = AssistantAgent(
    "ProblemAnalystAgent",
    description="A **strategic agent** that receives a business-focused task, performs a **root cause analysis**, and extracts 3-5 core, highly specific **technical keywords** and **research topics** essential for an ArXiv search.",
    model_client=model_client,
    system_message="""
    You are the Problem Strategist. Your role is to carefully analyze the user's business problem and transform it into a list of **3 to 5 highly specific, technical, and relevant search queries/keywords** for the PaperScoutingAgent. **Do not provide any analysis, conclusion, or summary of the papers.** Your response must be **only the keywords**, formatted as a numbered or bulleted list.

    **MANDATE:** After generating the keywords, you must end your message with a direct call for the next agent, like: "Keywords generated. @PaperScoutingAgent, please run the arxiv_tool with these terms."
    """
)

paper_scout = AssistantAgent(
    "PaperScoutingAgent",
    description="A **specialist researcher** who receives technical keywords and **executes the** `arxiv_tool` to fetch a curated list of research findings.",
    model_client=model_client,
    tools=[arxiv_tool],
    system_message="""
    **You are the Research Specialist.** Your sole function is to **use the available** `arxiv_tool` immediately upon receiving search terms from the ProblemAnalystAgent. You must formulate the best possible single query from the input and pass it to the tool. **Do not analyze, summarize, or reformat the findings.** Your only output should be the **raw results** returned by the tool, which you will pass directly to the ReportGenerator.

    **MANDATE:** After generating the findings, you must end your message with a direct call for the next agent, like: "Findings generated. @ReportGenerator, please create a report from findings."
    """
)

report_generator = AssistantAgent(
    "ReportGenerator",
    description="An **executive communicator** who synthesizes raw research findings into a **clear, structured, and professional technical brief** suitable for a business audience.",
    model_client=model_client,
    system_message="**You are the Executive Report Generator.** Your task is to take the raw paper data (which may contain up to 10 papers) and perform a **relevance assessment** against the original business problem. **Synthesize the findings from the TOP 5 most relevant papers** into a coherent, business-focused **Technical Brief** using professional **Markdown** formatting. The report must include: 1) A brief executive summary, 2) The key research papers found (Title, Authors, Abstract Summary) for the **top 5 selected papers**, and 3) A conclusion on the AI methodology trends based on the selected papers. **After generating the complete, final report, and only then, your last line must be the termination signal: LEET.**"
)

termination_cond = TextMentionTermination("LEET")

selector_prompt=selector_prompt="""
    You are the workflow coordinator. Your task is to select the **ABSOLUTELY CORRECT NEXT AGENT** from the list of candidates based on the current conversation history. Your selection must ensure the following **STRICT, THREE-STAGE WORKFLOW** is executed:

    1. **ProblemAnalystAgent** (Keywords Generation)
    2. **PaperScoutingAgent** (Tool Execution)
    3. **ReportGenerator** (Final Report Synthesis)

    ---

    **CURRENT CONTEXT:**
    {roles}
    
    **CONVERSATION HISTORY (Check the last message):**
    {history}

    ---

    **SELECTION LOGIC (Non-Negotiable):**

    * **INITIAL STEP:** If the conversation just started (or if the last message was from the user), select the **ProblemAnalystAgent**.
    * **AFTER KEYWORDS:** If the last message was from the **ProblemAnalystAgent** (i.e., keywords were generated), you **MUST** select the **PaperScoutingAgent**.
    * **AFTER TOOL CALL:** If the last action was the **PaperScoutingAgent** executing the `arxiv_tool` and the raw results were returned, you **MUST** select the **ReportGenerator**.
    * **NEVER:** **Do not select the ProblemAnalystAgent after the first step is complete.**

    Select only one agent from {participants} to perform the next required step in the sequence.
"""

# Creating the group chat
mvp_team = SelectorGroupChat(
    participants=[problem_analyst, paper_scout, report_generator],
    model_client=model_client,
    termination_condition=termination_cond,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=False
)

# Run the complete workflow -
async def create_technical_brief():
    task="""
    ## TASK GOAL: TECHNICAL BRIEF GENERATION

    **CONTEXT:** You are initiating a specialized research workflow designed to rapidly translate a business need into actionable technical findings using academic research. The team's output will directly inform a strategic decision on adopting AI methodologies.

    **BUSINESS PROBLEM:** Create a comprehensive **Technical Brief** focused on methodologies for **fraudulent transaction detection** using Artificial Intelligence.

    ---

    **WORKFLOW CONSTRAINTS & EXECUTION MANDATES:**

    1.  **Strict Sequencing:** The three-agent team must execute its mandated roles in sequential order: **ProblemAnalystAgent** $\rightarrow$ **PaperScoutingAgent** $\rightarrow$ **ReportGenerator**.
    2.  **Tool Use Mandate:** The PaperScoutingAgent **must** execute the `arxiv_tool` immediately upon receiving keywords from the Analyst.
    3.  **Data Focus:** The ReportGenerator must use the raw data from the top 10 papers fetched by the tool to identify and synthesize the **TOP 5 most relevant papers** for the final report.

    ---

    **EXPECTED FINAL OUTPUT STRUCTURE (FROM ReportGenerator):**

    The final output must be a professional technical brief, entirely in Markdown, featuring these sections:

    1.  **Executive Summary:** A concise, 2-3 paragraph overview of the current state of AI for fraud detection based on the top 5 papers.
    2.  **Key Research Findings (Top 5 Papers):** Detailed summaries for the five most relevant papers, including Title, Key Methodology, Authors, and a link (URL).
    3.  **Conclusion & Trend Analysis:** A brief analysis of the prevailing AI models identified in the top papers, and their applicability to the business problem.

    Begin the workflow now by selecting the **ProblemAnalystAgent** to define the key research areas for "AI for fraudulent transaction detection".
    """

    await Console(mvp_team.run_stream(task=task))

    # for msg in result.messages:
    #     print(f"{msg.source}: \n{msg.content}\n\n")
    
    await model_client.close()

asyncio.run(create_technical_brief())