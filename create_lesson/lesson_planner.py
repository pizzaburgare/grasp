import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage  # type: ignore
from langchain_core.tools import tool  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from pydantic import SecretStr

load_dotenv()

# --- Configuration ---
COURSE = "FMNF05"
TOPIC = "LU decomposition"
MODEL = "liquid/lfm-2.5-1.2b-thinking:free"

# 1. Define the LLM
llm = ChatOpenAI(
    model=MODEL,
    api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Math Lesson Agent",
    },
)


# --- Tools ---
@tool
def rag_search_in_exams(query: str) -> str:
    """Search previous FMNF05 exams for relevant LU decomposition problems."""
    return "Found 3 problems involving pivoting in LU decomposition from 2023 exam."


tools = [rag_search_in_exams]

# 3. Setup the System Prompt
with open("create_lesson/prompt.md", "r") as f:
    system_content = f.read().replace("<topic>", TOPIC)

# 4. Create the Agent
# langgraph's create_react_agent is the modern way to handle tool-calling loops
agent_executor = create_agent(llm, system_prompt=system_content)

# 5. Execution

messages = [SystemMessage(content=system_content)]

response = agent_executor.invoke({"messages": messages})  # type: ignore

# Print the last message in the conversation (the result)
print(response["messages"][-1].content)
