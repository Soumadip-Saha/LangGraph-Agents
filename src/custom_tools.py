from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

class StreamingInput(BaseModel):
    person_name: str = Field(..., description="The name of the person to greet")

class StreamingTool(BaseTool):
    name: str = "StreamingTool"
    description: str = "Useful to greet a person and encourage them"
    args_schema: Type[BaseModel] = StreamingInput
    
    def _run(self, person_name: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return f"Hello {person_name}"
    