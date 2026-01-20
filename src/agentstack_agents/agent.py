# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
import re
import json
from typing import Annotated
from textwrap import dedent

from dotenv import load_dotenv
from a2a.types import AgentSkill, Message, TextPart
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from agentstack_sdk.a2a.types import AgentMessage, AgentArtifact
from agentstack_sdk.a2a.extensions import (
    AgentDetail, AgentDetailTool,
    ErrorExtensionParams, ErrorExtensionServer, ErrorExtensionSpec,
    TrajectoryExtensionServer, TrajectoryExtensionSpec,
    LLMServiceExtensionServer, LLMServiceExtensionSpec
)
from agentstack_sdk.a2a.extensions.ui.canvas import CanvasExtensionServer, CanvasExtensionSpec
import httpx

load_dotenv()
server = Server()

def extract_code_blocks(text: str) -> list[dict]:
    """Extract code blocks with language and content"""
    pattern = r"```(\w+)?\n(.*?)```"
    blocks = []
    
    for match in re.finditer(pattern, text, re.DOTALL):
        language = match.group(1) or "text"
        code = match.group(2).strip()
        blocks.append({
            "language": language,
            "code": code,
            "start": match.start(),
            "end": match.end()
        })
    
    return blocks

@server.agent(
    name="Coding with Canvas",
    default_input_modes=["text", "text/plain"],
    default_output_modes=["text", "text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Hi! I'm a coding assistant. Ask me to write, explain, or edit code in any language.",
        tools=[
            AgentDetailTool(
                name="Canvas",
                description="Interactive code editing with selection-based modifications."
            )
        ],
        author={"name": "Jenna Winkler"},
        source_code_url="https://github.com/jenna-winkler/agentstack-coding-agent"
    ),
    skills=[
        AgentSkill(
            id="coding-agent-canvas",
            name="Coding Agent with Canvas",
            description=dedent(
                """\
                An AI-powered coding assistant that writes, explains, debugs, and edits code across multiple programming languages.
                Supports interactive canvas-based editing for precise code modifications.
                """
            ),
            tags=["Coding", "Canvas", "Development"],
            examples=[
                "Write a Python function to calculate fibonacci numbers with error handling",
                "Create a React component for a user login form with validation",
                "Build a JavaScript debounce function with configurable delay",
            ]
        )
    ],
)
async def coding_agent(
    input: Message,
    context: RunContext,
    canvas: Annotated[CanvasExtensionServer, CanvasExtensionSpec()],
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(
            suggested=("ibm-granite/granite-4.0-h-small",)
        )
    ],
    _e: Annotated[ErrorExtensionServer, ErrorExtensionSpec(ErrorExtensionParams(include_stacktrace=True))],
):
    """A coding assistant that generates, explains, and edits code with canvas support."""

    await context.store(input)

    yield trajectory.trajectory_metadata(title="Initializing", content="Starting coding agent")

    canvas_edit_request = await canvas.parse_canvas_edit_request(message=input)
    
    user_msg = ""
    for part in input.parts:
        if part.root.kind == "text":
            user_msg = part.root.text
            break

    if canvas_edit_request:
        original_code = (
            canvas_edit_request.artifact.parts[0].root.text
            if isinstance(canvas_edit_request.artifact.parts[0].root, TextPart)
            else ""
        )
        selected_code = original_code[canvas_edit_request.start_index:canvas_edit_request.end_index]
        
        prompt = f"""Edit this code:

SELECTED:
```
{selected_code}
```

FULL CODE:
```
{original_code}
```

USER REQUEST: {user_msg}

Return ONLY the complete updated code in a code block (```language)."""

        yield trajectory.trajectory_metadata(
            title="Canvas Edit",
            content=f"Editing {len(selected_code)} chars"
        )
    else:
        prompt = user_msg

    try:
        if not llm or not llm.data or not llm.data.llm_fulfillments:
            error_msg = "LLM service not available"
            msg = AgentMessage(text=error_msg)
            yield msg
            await context.store(msg)
            return

        llm_config = llm.data.llm_fulfillments.get("default")
        if not llm_config:
            error_msg = "LLM config not found"
            msg = AgentMessage(text=error_msg)
            yield msg
            await context.store(msg)
            return

        system_prompt = """You are an expert coding assistant.

Write clean, well-documented code with proper error handling.
ALWAYS wrap code in markdown code blocks with language: ```python, ```javascript, etc.
For edits, return the COMPLETE updated code.
Keep explanations brief."""

        yield trajectory.trajectory_metadata(title="Processing", content=f"Using {llm_config.api_model}")

        # Build the full URL properly
        base_url = llm_config.api_base.rstrip('/')
        full_url = f"{base_url}/chat/completions"
        
        yield trajectory.trajectory_metadata(title="Debug URL", content=f"Calling: {full_url}")

        request_data = {
            "model": llm_config.api_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }

        response_text = ""
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {llm_config.api_key}"
            }
            
            async with client.stream(
                "POST",
                full_url,
                json=request_data,
                headers=headers
            ) as response:
                yield trajectory.trajectory_metadata(title="Debug Status", content=f"HTTP {response.status_code}")
                
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                        
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        if chunk_data.get("choices") and len(chunk_data["choices"]) > 0:
                            delta = chunk_data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                response_text += content
                                yield content
                    except json.JSONDecodeError:
                        continue

        code_blocks = extract_code_blocks(response_text)

        if code_blocks:
            primary_block = max(code_blocks, key=lambda b: len(b["code"]))
            
            artifact_name = f"{primary_block['language'].title()} Code"
            
            first_lines = primary_block["code"].split("\n")[:3]
            for line in first_lines:
                if match := re.match(r"^\s*[#//]\s*(.+)$", line):
                    title = match.group(1).strip()
                    if len(title) < 50 and not title.lower().startswith("copyright"):
                        artifact_name = title
                        break

            code_with_formatting = f"```{primary_block['language']}\n{primary_block['code']}\n```"

            artifact = AgentArtifact(
                name=artifact_name,
                parts=[TextPart(text=code_with_formatting)],
            )
            
            yield trajectory.trajectory_metadata(
                title="Code Generated",
                content=f"{primary_block['language']} ({len(primary_block['code'])} chars)"
            )

            yield artifact
            await context.store(artifact)

        msg = AgentMessage(text=response_text)
        await context.store(msg)

        yield trajectory.trajectory_metadata(title="Complete", content="Done")

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        yield trajectory.trajectory_metadata(title="Error", content=error_msg)
        msg = AgentMessage(text=error_msg)
        yield msg
        await context.store(msg)

def run():
    server.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8008)),
        configure_telemetry=True,
        context_store=PlatformContextStore()
    )

if __name__ == "__main__":
    run()