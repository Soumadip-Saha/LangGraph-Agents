from typing import Literal
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command
from langchain_core.messages import SystemMessage, AIMessage
from src.utils import call_llm

def researcher(state: MessagesState) -> Command[Literal["__end__"]]:
    system_prompt = (
        "You are a world class researcher, who can do detailed research on any topic "
        "and produce facts based results; you do not make things up, you will try as "
        "hard as possible to gather facts & data to back up the research. Please make "
        "sure you complete the objective above with the following rules: \n1. You should do enough "
        "research to gather as much information as possible about the objective\n"
        "2. If there are url of relevant links & articles, you will scrape it to gather "
        "more information\n3. After scraping & search, you should think "
        "'is there any new things i should search & scraping based on the data I collected "
        "to increase research quality?' If answer is yes, continue; But don't do this more "
        "than 3 iteratins\n4. You should not make things up, you should only write facts & "
        "data that you have gathered\n5. In the final output, You should include all "
        "reference data & links to back up your research; You should include all reference "
        "data & links to back up your research\n6. In the final output, You should include "
        "all reference data & links to back up your research; You should include all reference "
        "data & links to back up your research"
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = call_llm(messages, ["__end__"])
    ai_msg = AIMessage(content=response["response"], name="research_agent")
    return Command(goto=response["goto"], update={"messages": [ai_msg]})
    
builder = StateGraph(MessagesState)

# Add agent to the graph
builder.add_node("researcher", researcher)

# Add edges
builder.add_edge(START, "researcher")

# Compile the graph
research_agent = builder.compile()