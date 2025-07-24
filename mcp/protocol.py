from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import uuid
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

class MCPMessage(BaseModel):
    """Model Context Protocol (MCP) message structure."""
    sender: str = Field(..., description="Sender of the message")
    receiver: str = Field(..., description="Receiver of the message")
    type: str = Field(..., description="Type of the message")
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Trace ID for tracking message flow")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Message payload")
    
    def to_json(self) -> str:
        """Convert the message to JSON string."""
        return json.dumps(self.model_dump())
    
    @classmethod
    def from_json(cls, json_str: str) -> "MCPMessage":
        """Create a message from JSON string."""
        data = json.loads(json_str)
        return cls(**data)
