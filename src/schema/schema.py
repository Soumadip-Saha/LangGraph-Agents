from typing import Any
from pydantic import BaseModel, Field, SerializeAsAny
from schema.models import AllModelEnum, OpenAIModelName
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
    
    

class AgentInfo(BaseModel):
    """Info about an available agent."""
    
    key: str = Field(
        description="Agent key.",
        examples=["research-agent"],
    )
    
    description: str = Field(
        description="Description of the agent.",
        examples=["Agent for research"],
    )
    
class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""
    
    agents: list[AgentInfo] = Field(
        description="List of available agents."
    )
    models: list[AllModelEnum] = Field(
        description="List of all available LLMs."
    )
    default_agent: str = Field(
        description="Default agent when none is specified.",
        examples=["research-agent"]
    )
    default_model: AllModelEnum = Field(
        description="Default model when none is specified.",
    )

class UserInput(BaseModel):
    """Basic user input for the agent."""
    
    message: str = Field(
        description="User input to the agent.",
        examples=["What is the capital of India?"]
    )
    model: SerializeAsAny[AllModelEnum] | None = Field(
        title="Model",
        description="LLM model to use for the agent.",
        default=OpenAIModelName.GPT_4O_MINI,
        examples=[OpenAIModelName.GPT_4O_MINI]
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-tern conversation.",
        default=None,
        examples=["25435-q59t957-q9tq3t8q9-5t47q5yy"]
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration for the agent.",
        default={},
        examples=[{"foo": "bar"}]
    )

class StreamInput(UserInput):
    """User input for streaming the agent's response."""
    
    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True
    )
    
class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""
    
    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-tern conversation.",
        examples=["25435-q59t957-q9tq3t8q9-5t47q5yy"]
    )

class ChatHistory(BaseModel):
    
    messages: list[ChatMessage]