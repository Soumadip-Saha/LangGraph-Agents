from fastapi import Request
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from schema import ChatMessage

def get_client_ip(request: Request) -> str:
    """Get the client IP address from the request headers."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Get the first IP address in the list
        return forwarded_for.split(",")[0].strip()
    
    # Fallback to the client's IP address
    return request.client.host if request.client else "unknown"

def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """Create a ChatMessage from a LangChain message."""
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content)
            )
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content)
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id
            )
            return tool_message
        # TODO: Add custom message types [https://github.com/JoshuaC215/agent-service-toolkit/blob/5db28a3dd1dcc15758a020848c061b0e01fbc67c/src/service/utils.py#L53]
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")

def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_call"
    ]

def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["content"])
    return "".join(text)
