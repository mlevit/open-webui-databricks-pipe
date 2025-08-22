# Databricks LLM Pipe for Open WebUI

[![GitHub Repository](https://img.shields.io/badge/GitHub-mlevit%2Fopen--webui--databricks--pipe-blue?logo=github)](https://github.com/mlevit/open-webui-databricks-pipe)

A streamlined integration pipe that connects Open WebUI with Databricks LLM serving endpoints, supporting both Personal Access Token and OAuth authentication methods.

## Features

- 🚀 **Auto-discovery**: Automatically finds available LLM endpoints from your Databricks workspace
- 🔍 **Smart filtering**: Only shows endpoints with specific allowed task types
- 🎯 **Simple naming**: Uses endpoint names exactly as they appear in Databricks
- 🔄 **OpenAI compatible**: Provides OpenAI-style API responses for seamless integration
- ⚡ **Streaming support**: Real-time response streaming for better user experience
- 🔐 **Dual authentication**: Supports both Personal Access Token and OAuth (Service Principal) authentication
- 🔄 **Token management**: Automatic OAuth token refresh with expiration handling

## Configuration

### Authentication Methods

This pipe supports two authentication methods:

#### Personal Access Token (Recommended for individual use)

- **Databricks Host**: Your workspace URL (e.g., `your-workspace.cloud.databricks.com`)
- **Authentication Method**: Select "Personal Access Token"
- **Personal Access Token**: Your Databricks personal access token

#### OAuth Service Principal (Recommended for production)

- **Databricks Host**: Your workspace URL (e.g., `your-workspace.cloud.databricks.com`)
- **Authentication Method**: Select "OAuth"
- **OAuth Client ID**: Your service principal's client ID
- **OAuth Secret**: Your service principal's secret

### Model Configuration

- **Auto-discover Models**: Enable/disable automatic endpoint discovery (default: enabled)
- **Custom Models** (optional): JSON array of custom models if you prefer manual configuration

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

### 1. Deploy the Pipe

Copy `databricks_pipe.py` to your Open WebUI pipes directory (typically `./data/pipes/`)

### 2. Configure Authentication

#### For Personal Access Token:

1. Generate a Databricks Personal Access Token:
   - Go to your Databricks workspace → User Settings → Developer → Access Tokens
   - Click "Generate New Token"
   - Copy the generated token

#### For OAuth (Service Principal):

1. Create a Service Principal in your Databricks workspace:
   - Go to Admin Settings → Identity and Access → Service Principals
   - Click "Add Service Principal"
   - Note the Application ID (Client ID)
2. Generate a secret for the Service Principal:
   - Click on your Service Principal → Secrets → Generate Secret
   - Copy the generated secret
3. Grant necessary permissions to the Service Principal for accessing serving endpoints

### 3. Configure in Open WebUI

1. Navigate to Admin Panel → Settings → Pipelines
2. Find "Databricks LLM Pipe" and configure the required fields
3. Test the connection - available models should appear automatically

## License

MIT License - see LICENSE file for details.
