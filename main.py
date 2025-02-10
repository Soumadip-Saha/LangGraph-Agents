import json
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.agents import research_agent
from langchain_core.messages import HumanMessage
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.schema import RequestFormatter
from src.utils import (
    get_client_ip, 
    langchain_to_chat_message, 
    convert_message_content_to_string, 
    remove_tool_calls
)
from langgraph.types import Command
# saved compiled graph

# with open("graph.png", "wb") as f:
#     f.write(research_agent.get_graph().draw_mermaid_png())

def setup_logging():
    """Configure logging with daily rotation and 7 days retention."""
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "api.log")
    handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    
    # Custom format including timestamp, level, IP address, and message
    formatter = RequestFormatter(
        "%(asctime)s - %(levelname)s - %(clientip)s - %(message)s"
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    logger.addHandler(handler)
    return logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    global logger
    logger = setup_logging()
    logger.info("Starting API server...", extra={"clientip": "system"})
    
    yield
    
    # Shutdown: Clean up logging handlers
    logger.info("Shutting down API server...", extra={"clientip": "system"})
    for handler in logger.handlers:
        handler.close()


class InputRequest(BaseModel):
    query: str

async def message_generator(query: str, client_ip: str):
    messages = [HumanMessage(content=query)]
    
    async for event in research_agent.astream_events({"messages": messages}, version="v2"):
        if not event:
            continue
        
        logger.info(
            event, extra={"clientip": client_ip}
        )
        
        new_messages = []
        # Yield messages written to the graph state after node execution finishes
        if (
            event["event"] == "on_chain_end"
            # on_chain_end gets called a bunch of times in a graph execution
            # Thif filters out everything except for "graph node finished"
            and any(t.startswith("graph:step:") for t in event.get("tags", []))
            and isinstance(event["data"]["output"], Command)
        ):
            new_messages = event["data"]["output"].update["messages"]
            
        # TODO: Add custom event yield logic [https://github.com/JoshuaC215/agent-service-toolkit/blob/5db28a3dd1dcc15758a020848c061b0e01fbc67c/src/service/service.py#L155]
        
        for message in new_messages:
            try:
                chat_message = langchain_to_chat_message(message)
                # TODO: Add run_id [https://github.com/JoshuaC215/agent-service-toolkit/blob/5db28a3dd1dcc15758a020848c061b0e01fbc67c/src/service/service.py#L162]
            except Exception as e:
                logger.error(f"Error parsing message: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                continue
            # LangGraph re-sends the input message, so drop it
            if chat_message.type == "human" and chat_message.content == query:
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"
        
        # Yield tokens streamed from LLMs
        if (
            event["event"] == "on_chat_model_stream"
            # TODO: Add option of streaming from UI [https://github.com/JoshuaC215/agent-service-toolkit/blob/5db28a3dd1dcc15758a020848c061b0e01fbc67c/src/service/service.py#L175]
        ):
            content = remove_tool_calls(event["data"]["chunk"].content)
            if content:
                # Empty content is the context of OpenAI usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content.
                yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
            continue
        
    
    yield "data: [DONE]\n\n"

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/stream")
async def stream(input_request: InputRequest, request: Request):
    client_ip = get_client_ip(request)
    return StreamingResponse(
        message_generator(input_request.query, client_ip), media_type="text/event-stream"
    )