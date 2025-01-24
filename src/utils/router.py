
import json
from langchain_core.messages import AnyMessage
from typing import List
from src.models import model

def call_llm(messages: List[AnyMessage], target_agent_nodes: List[str]):
    json_schema = {
        "name": "Response",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "A human readable response to the original question. Does not need to be a final response. Will be streamed back to the user. Unless goto is '__end__' this should not be the final response, rather should be the steps update.",
                },
                "goto": {
                    "enum": target_agent_nodes,
                    "type": "string",
                    "description": "The next agent to call, or __end__ if the user's query has been resolved. Must be one of the specified values.",
                }
            },
            "required": ["response", "goto"]
        }
    }
    response = model.with_structured_output(json_schema).invoke(messages)
    return response