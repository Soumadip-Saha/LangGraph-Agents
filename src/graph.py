from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langchain_core.messages import AIMessage
from time import sleep
from langchain_core.callbacks import CallbackManagerForToolRun
# from src.custom_tools import 

def greet_person(name: str):
    message = (
        f"Hello, {name}!\n"
        "I hope you're having an amazing day. Remember, every day is an opportunity to learn, grow, "
        "and achieve something new. Stay positive, keep chasing your dreams, and always believe in yourself. "
        "The world is full of possibilities, and you have the potential to make a real difference."
    )
    for text in message.split(" "):
        sleep(0.05)
        yield text

@tool
def tool_greet_person(name: str):
    """Generates a medium-length personalized greeting message."""
    message = ""
    for text in greet_person(name):
        print(text, end=" ", flush=True)
        message += f"{text} "
    return message.strip()
        

tools = [tool_greet_person]

tool_node = ToolNode(tools, messages_key="info")

message_to_use_tool = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "tool_greet_person",
            "args": {"name": "Alice"},
            "id": "tool_call_id",
            "type": "tool_call"
        }
    ]
)

tool_callback = CallbackManagerForToolRun()

output = tool_node.invoke(
    {"info": [message_to_use_tool], "config":{"callbacks": [tool_callback]}}
)

print("\n", output)

print(tool_callback)