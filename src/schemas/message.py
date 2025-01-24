from typing import Any, Literal, NotRequired
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class ToolCall(TypedDict):
    """Represents a request to call a tool."""
    
    name: str
    """Name of the tool"""
    args: dict[str, Any]
    """The arguments to the tool call"""
    id: str | None
    """An identifier associated with the tool call"""
    type: NotRequired[Literal["tool_call"]]

class ChatMessage(BaseModel):
    """Message in a chat"""
    
    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="Content of the message",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to",
        default=None,
        examples=["call_Jrjkn3g23nenvkjkserWIUHQIBFrej"],
    )
    run_id: str | None = Field(
        description="Run id of the message",
        default=None,
        examples=["run_Jrjkn3g23nenvkjkserWIUHQIBFrej"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token count, etc.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data",
        default={},
    )
    
    def pretty_repr(self) -> str:
        """Get a pretty representation of the message"""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"
    
    def pretty_print(self) -> None:
        """Pretty print the message"""
        print(self.pretty_repr())
    
    