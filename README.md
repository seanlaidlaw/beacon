# Beacon — Semantic Code Intelligence Engine

Beacon indexes your codebase with semantic understanding, enabling powerful search, context capsules, and MCP integration for AI assistants.

## Features

- **Semantic Search**: Find code by meaning, not just keywords
- **Context Capsules**: Generate concise, relevant context windows for AI agents
- **Call Graph & Impact Analysis**: Visualize dependencies and understand blast radius
- **Incremental Indexing**: Automatically re‑index only changed files
- **Multi‑Language Support**: Python, JavaScript/TypeScript, Go, Rust, Java, C/C++, Bash, Lua, Swift, R, Markdown, and more
- **MCP Server**: Built‑in Model Context Protocol server for AI coding assistants

## Quick Start

```bash
# Install Beacon via uv
uv tool install git+https://github.com/seanlaidlaw/beacon

# Setup hooks for automatic indexing
beacon setup

# Index your codebase
beacon index

# Now either:
# - Run Claude Code (auto-starts beacon mcp)
# - Or search manually with semantic search
beacon ask
```


## Commands

- `beacon setup` - Install project‑local hook and generate MCP configuration
- `beacon index` - Scan and index a codebase
- `beacon ask` - Interactive TUI for exploring code
- `beacon search <query>` - Hybrid semantic + keyword search
- `beacon capsule <query>` - Generate a context capsule for AI agents
- `beacon mcp` - Start the MCP server for integration with AI assistants

## MCP Integration

Beacon provides a Model Context Protocol server that lets AI assistants like Claude Code search your codebase, retrieve context capsules, and understand code relationships—all in real time.

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/beacon-code
cd beacon-code

# Install in development mode
pip install -e .
```
