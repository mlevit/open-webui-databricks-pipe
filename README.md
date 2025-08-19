# Databricks LLM Pipe for Open WebUI

[![GitHub Repository](https://img.shields.io/badge/GitHub-mlevit%2Fopen--webui--databricks--pipe-blue?logo=github)](https://github.com/mlevit/open-webui-databricks-pipe)

A streamlined integration pipe that connects Open WebUI with Databricks LLM serving endpoints.

## Features

- 🚀 **Auto-discovery**: Automatically finds available LLM endpoints from your Databricks workspace
- 🔍 **Smart filtering**: Only shows endpoints with specific allowed task types
- 🎯 **Simple naming**: Uses endpoint names exactly as they appear in Databricks
- 🔄 **OpenAI compatible**: Provides OpenAI-style API responses for seamless integration
- ⚡ **Streaming support**: Real-time response streaming for better user experience

## Configuration

Set these required parameters in Open WebUI:

- **Databricks Host**: Your workspace URL (e.g., `your-workspace.cloud.databricks.com`)
- **Databricks Token**: Your Databricks personal access token
- **Custom Models** (optional): JSON array of custom models if you prefer manual configuration
- **Auto-discover Models**: Enable/disable automatic endpoint discovery (default: enabled)

### Custom Models Format

If you prefer to manually specify models instead of auto-discovery, use this JSON format:

```json
[
  {
    "id": "my-llama-name",
    "name": "My Llama Model"
  },
  {
    "id": "my-mixtral-name",
    "name": "My Mixtral Model"
  }
]
```

Where:

- `id`: The exact Databricks endpoint name
- `name`: The display name shown in Open WebUI

## How it Works

The pipe automatically discovers Databricks endpoints that are:

- In `READY` state
- Have a `task` field matching one of the allowed endpoint types:
  - `agent/v1/chat`
  - `agent/v1/responses`
  - `agent/v2/chat`
  - `llm/v1/chat`
  - `llm/v1/completions`

## Installation

1. Copy `databricks_pipe.py` to your Open WebUI pipes directory
2. Configure your Databricks credentials
3. The pipe will automatically discover and list available LLM endpoints

## Requirements

- `requests`
- `pydantic`

## License

MIT License - see LICENSE file for details.
