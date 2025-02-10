import datetime
from typing import Literal
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableSerializable, RunnableLambda, RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.tools import TavilySearchResults

from core import get_model

tools = [
    TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True
    )
]

system_template = PromptTemplate.from_template(
    (
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
        "data & links to back up your research\n\n"
        "For your information, today's date is {date}.\n\n"
    )
)

def wrap_model(model: BaseChatModel) -> RunnableSerializable[MessagesState, AIMessage]:
    if len(tools)>0:
        model = model.bind_tools(tools)
    instructions = system_template.invoke({"date": datetime.datetime.now().strftime("%B %d, %Y")}).to_string()
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier"
    )
    return preprocessor | model

async def acall_model(state: MessagesState, config: RunnableConfig) -> MessagesState:
    m = get_model(config["configurable"].get("model", "gpt-4o-mini"))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)
    
    return {"messages": [response]}

# Define the graph
agent = StateGraph(MessagesState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.set_entry_point("model")

agent.add_edge("tools", "model")

def call_tool(state: MessagesState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"

agent.add_conditional_edges("model", call_tool, {"tools": "tools", "done": "__end__"})

research_agent = agent.compile(checkpointer=MemorySaver())