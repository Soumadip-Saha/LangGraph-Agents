from dataclasses import dataclass
import os
from langgraph.graph.state import CompiledStateGraph
from agents.research_agent import research_agent
from schema import AgentInfo

# Save image
current_dir = os.getcwd()
file_path = os.path.join(current_dir, "graph.png")
with open(file_path, "wb") as f:
    f.write(research_agent.get_graph().draw_mermaid_png())
    print(f"Graph saved to {file_path}")

DEFAULT_AGENT = "research-agent"

@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph
    
agents: dict[str, Agent] = {
    "research-agent": Agent(description="Agent for research", graph=research_agent)
}

def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph

def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description)
        for agent_id, agent in agents.items()
    ]