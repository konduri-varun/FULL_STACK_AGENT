# In LLM agents, agents use LLMs for reasoning and decision making

# Import required ADK components
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools.agent_tool import AgentTool

# ============================================================================
# HIERARCHICAL MULTI-AGENT SYSTEM
# ============================================================================
# This system uses a hierarchical architecture where a root agent delegates
# tasks to specialized sub-agents based on the type of work required.
#
# Architecture:
#   root_agent (Manager) → Delegates to → search_agent or code_agent (Workers)
# ============================================================================

# Create specialized agents for each tool
# SUB-AGENT 1: Web Search Specialist
search_agent = Agent(
    model="gemini-2.0-flash",                    # LLM for reasoning
    name="search_agent",                         # Unique identifier
    description="Specialist in web search using Google Search.",  # What it does
    instruction="Use the Google Search tool to find current, accurate information from the web.",  # How to behave
    tools=[google_search],                       # Available tools
)

# SUB-AGENT 2: Code Execution Specialist
code_agent = Agent(
    model="gemini-2.0-flash",                    # LLM for reasoning
    name="code_agent",                           # Unique identifier
    description="Specialist in executing Python code for calculations and data analysis.",  # What it does
    instruction="""You are a code execution specialist. When given a computational task:

1. Write clean, correct Python code to solve the problem
2. Execute the code using your built-in code execution capability
3. Return only the final result as plain text, without code blocks or markdown
4. For calculations, provide just the numerical answer
5. For data analysis, provide clear, concise results

Example: If asked "What is 5 + 7?", write and execute code that prints "12".""",  # Detailed behavior guide
    code_executor=BuiltInCodeExecutor(),         # Code execution capability
)

# ROOT AGENT: Task Coordinator and Delegator
# This is the main agent that users interact with. It analyzes requests and
# delegates to the appropriate specialized sub-agents.
root_agent = Agent(
    model="gemini-2.0-flash",                    # LLM for intelligent delegation
    name="my_first_agent",                       # System name
    description="A versatile AI agent that can search the web and execute code through specialized sub-agents.",  # Overall capability
    instruction="""You are a helpful AI assistant with access to web search and code execution capabilities through specialized agents.

When answering questions:
1. For web searches and current information: delegate to the search_agent
2. For calculations, data analysis, or code execution: delegate to the code_agent
3. Provide clear, well-structured responses with explanations
4. Always prioritize accuracy and be helpful to the user

Delegate tasks to the appropriate specialized agent based on the user's needs.""",  # Delegation logic
    tools=[
        AgentTool(agent=search_agent),           # Web search capability via delegation
        AgentTool(agent=code_agent),             # Code execution capability via delegation
    ],
)