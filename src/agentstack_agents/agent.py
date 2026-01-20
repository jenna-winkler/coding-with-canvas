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
    name="Coding Agent with Canvas",
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

        base_url = llm_config.api_base.rstrip('/')
        full_url = f"{base_url}/chat/completions"

        request_data = {
            "model": llm_config.api_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }

        response_text = ""
        in_code_block = False
        code_buffer = ""
        language = ""
        artifact_id = None
        pre_code_text = ""
        
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
                                
                                # Check for code block start
                                if not in_code_block and "```" in content:
                                    # Extract language and start code block
                                    parts = content.split("```", 1)
                                    pre_code_text += parts[0]
                                    if len(parts) > 1:
                                        # Get language from first line
                                        remaining = parts[1]
                                        if '\n' in remaining:
                                            lang_line, code_start = remaining.split('\n', 1)
                                            language = lang_line.strip() or "text"
                                            code_buffer = code_start
                                        else:
                                            language = remaining.strip() or "text"
                                            code_buffer = ""
                                        in_code_block = True
                                        
                                        # Create first artifact chunk
                                        artifact = AgentArtifact(
                                            name=f"{language.title()} Code",
                                            parts=[TextPart(text=f"```{language}\n{code_buffer}")]
                                        )
                                        artifact_id = artifact.artifact_id
                                        yield artifact
                                        await context.store(artifact)
                                elif in_code_block:
                                    # Check if code block ends
                                    if "```" in content:
                                        parts = content.split("```", 1)
                                        code_buffer += parts[0]
                                        in_code_block = False
                                        
                                        # Send final artifact chunk
                                        final_artifact = AgentArtifact(
                                            artifact_id=artifact_id,
                                            name=f"{language.title()} Code",
                                            parts=[TextPart(text=parts[0])]
                                        )
                                        yield final_artifact
                                    else:
                                        # Continue code block
                                        code_buffer += content
                                        chunk_artifact = AgentArtifact(
                                            artifact_id=artifact_id,
                                            name=f"{language.title()} Code",
                                            parts=[TextPart(text=content)]
                                        )
                                        yield chunk_artifact
                                else:
                                    # Regular text, just stream it
                                    yield content
                                    
                    except json.JSONDecodeError:
                        continue

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