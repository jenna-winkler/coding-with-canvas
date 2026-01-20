# Coding Agent with Canvas

A simple AI coding assistant that generates code and supports interactive canvas-based editing.

## Features

- **Code Generation** - Write code in any programming language
- **Canvas Editing** - Select and edit specific parts of generated code
- **Streaming** - Watch code appear in real-time
- **Multi-language** - Supports Python, JavaScript, React, and more

## Installation

```bash
git clone https://github.com/jenna-winkler/agentstack-coding-agent
cd agentstack-coding-agent
uv sync
```

## Usage

Start the agent:

```bash
uv run server
```

The agent will be available at `http://127.0.0.1:8008`

## Example Prompts

- "Write a Python function to calculate fibonacci numbers with error handling"
- "Create a React component for a user login form with validation"
- "Build a JavaScript debounce function with configurable delay"

## How Canvas Works

1. Generate code with any prompt
2. Click and drag to select part of the code in the canvas
3. Ask the agent to modify just that section
4. The agent returns the complete updated code

## Requirements

- Python 3.11+
- Agent Stack SDK 0.5.2rc2+
- Access to an LLM provider (configured via Agent Stack platform)
