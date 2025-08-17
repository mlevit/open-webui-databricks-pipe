"""
Title: Databricks LLM Pipe (Open WebUI)
Version: 0.1.0
"""

import requests
import json
import time
import uuid
from typing import List, Union, Generator, Dict, Any
from pydantic import BaseModel, Field


class Pipe:
    """
    Databricks LLM integration pipe optimized for Open WebUI.
    """

    class Valves(BaseModel):
        """Configuration model with validation and sensible defaults."""

        databricks_host: str = Field(
            default="your-workspace.cloud.databricks.com",
            description="Databricks host name (required)",
        )
        databricks_token: str = Field(
            default="dapixxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            description="Databricks authentication token (required)",
        )
        custom_models: str = Field(
            default="",
            description="Custom models (JSON format: [{'id': 'endpoint-name', 'name': 'Display Name'}]). Leave empty to use auto-discovery.",
        )
        auto_discover_models: bool = Field(
            default=True,
            description="Automatically discover available LLM serving endpoints",
        )

    # No constants needed for simplified approach

    # Fallback models if auto-discovery fails and no custom models provided
    DEFAULT_MODELS = [
        {
            "id": "databricks-meta-llama-3-1-70b-instruct",
            "name": "Meta Llama 3.1 70B Instruct",
        },
    ]

    def __init__(self):
        self.name = ""
        self.valves = self.Valves()
        self._cached_models = None
        self._cache_timestamp = 0
        self._cache_duration = 300  # 5 minutes

    def _validate_config(self) -> Union[str, None]:
        """Validate configuration. Returns error message or None if valid."""
        if not self.valves.databricks_host.strip():
            return "Databricks host is required"

        if not self.valves.databricks_token.strip():
            return "Databricks token is required"

        return None

    def _get_headers(self) -> Dict[str, str]:
        """Generate request headers."""
        return {
            "Authorization": f"Bearer {self.valves.databricks_token.strip()}",
            "Content-Type": "application/json",
        }

    def _parse_custom_models(self) -> List[Dict[str, str]]:
        """Parse custom models from valves configuration."""
        if not self.valves.custom_models.strip():
            return []

        try:
            custom_models = json.loads(self.valves.custom_models.strip())
            if not isinstance(custom_models, list):
                return []

            return [
                {"id": str(model["id"]).strip(), "name": str(model["name"]).strip()}
                for model in custom_models
                if isinstance(model, dict) and "id" in model and "name" in model
            ]
        except json.JSONDecodeError:
            return []

    def _is_llm_endpoint(self, endpoint: Dict[str, Any]) -> bool:
        """Check if an endpoint is an LLM by looking at task field or endpoint_type."""
        # Check if endpoint_type is FOUNDATION_MODEL_API
        if endpoint.get("endpoint_type") == "FOUNDATION_MODEL_API":
            return True

        # Check if task contains 'llm' or 'agent' at endpoint level
        task = endpoint.get("task", "").lower()
        return "llm" in task or "agent" in task

    def _discover_models(self) -> List[Dict[str, str]]:
        """Auto-discover available LLM serving endpoints from Databricks."""
        if not self.valves.auto_discover_models or self._validate_config():
            return []

        try:
            response = requests.get(
                f"https://{self.valves.databricks_host.strip()}/api/2.0/serving-endpoints",
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()

            return [
                {"id": endpoint["name"], "name": endpoint["name"]}
                for endpoint in response.json().get("endpoints", [])
                if endpoint.get("name")
                and endpoint.get("state", {}).get("ready") == "READY"
                and self._is_llm_endpoint(endpoint)
            ]
        except Exception:
            return []  # Silently fail, fall back to other methods

    def _get_models(self) -> List[Dict[str, str]]:
        """Get models using priority: custom -> auto-discovery -> defaults."""
        # Check cache first
        current_time = time.time()
        if (
            self._cached_models is not None
            and current_time - self._cache_timestamp < self._cache_duration
        ):
            return self._cached_models

        models = []

        # Priority 1: Custom models from configuration
        custom_models = self._parse_custom_models()
        if custom_models:
            models = custom_models
        else:
            # Priority 2: Auto-discovered models
            discovered_models = self._discover_models()
            if discovered_models:
                models = discovered_models
            else:
                # Priority 3: Default fallback models
                models = self.DEFAULT_MODELS.copy()

        # Cache the result
        self._cached_models = models
        self._cache_timestamp = current_time

        return models

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> bool:
        """Validate message format."""
        if not isinstance(messages, list) or not messages:
            return False

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ["system", "user", "assistant", "function"]:
                return False

        return True

    def pipes(self) -> List[Dict[str, Any]]:
        """Return available model configurations."""
        current_time = int(time.time())

        return [
            {
                "id": model["id"],
                "name": model["name"],
                "object": "model",
                "created": current_time,
                "owned_by": "databricks",
            }
            for model in self._get_models()
        ]

    def pipe(
        self, body: Dict[str, Any]
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Main pipe method to handle both streaming and non-streaming requests.

        Args:
            body: Request body containing messages and parameters

        Returns:
            Response dictionary or generator for streaming responses
        """
        # Validate configuration
        config_error = self._validate_config()
        if config_error:
            return {"error": f"Configuration error: {config_error}"}

        messages = body.get("messages", [])

        # Validate messages
        if not self._validate_messages(messages):
            return {
                "error": "Invalid messages format. Messages must be a non-empty list with 'role' and 'content' keys."
            }

        # Prepare payload
        payload = self._prepare_payload(body)
        headers = self._get_headers()

        try:
            if body.get("stream", False):
                return self._stream_response(payload, headers)
            else:
                return self._non_stream_response(payload, headers)
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

    def _prepare_payload(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request payload from body."""
        # Start with essential parameters
        payload = {
            "messages": body.get("messages", []),
            "stream": body.get("stream", False),
        }

        # Add model if specified and strip pipe prefix
        if model := body.get("model"):
            payload["model"] = model[model.find(".") + 1 :]

        # Add all other parameters, excluding already handled ones
        excluded_params = {"messages", "stream", "model"}
        payload.update(
            {
                param: value
                for param, value in body.items()
                if param not in excluded_params and value is not None
            }
        )

        return payload

    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request."""
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def _stream_response(
        self, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Generator[str, None, None]:
        """Handle streaming response."""
        payload["stream"] = True
        url = f"https://{self.valves.databricks_host.strip()}/serving-endpoints/{payload.get('model', '')}/invocations"

        try:
            response = self._make_request(
                "POST", url, headers=headers, json=payload, stream=True
            )

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8").strip()

                # Handle Server-Sent Events format
                if line.startswith("data: "):
                    data_content = line[6:].strip()

                    if data_content == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break

                    try:
                        chunk = json.loads(data_content)
                        processed_chunk = self._process_stream_chunk(chunk)
                        if processed_chunk:
                            yield f"data: {json.dumps(processed_chunk)}\n\n"
                    except json.JSONDecodeError:
                        # Skip malformed chunks
                        continue

        except Exception as e:
            error_response = self._handle_request_error(e)
            yield f"data: {json.dumps(error_response)}\n\n"

    def _process_stream_chunk(
        self, chunk: Dict[str, Any]
    ) -> Union[Dict[str, Any], None]:
        """Process individual streaming chunk."""
        if not isinstance(chunk, dict) or "choices" not in chunk:
            return None

        choices = chunk.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})

        # Only process chunks with content
        if "content" not in delta:
            return None

        return {
            "id": chunk.get("id", f"chatcmpl-{uuid.uuid4()}"),
            "object": "chat.completion.chunk",
            "created": chunk.get("created", int(time.time())),
            "model": chunk.get("model", "databricks-model"),
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "delta": {"content": delta["content"]},
                    "finish_reason": choice.get("finish_reason"),
                }
            ],
        }

    def _handle_request_error(self, e: Exception) -> Dict[str, str]:
        """Handle different types of request errors consistently."""
        if isinstance(e, requests.exceptions.Timeout):
            return {"error": "Request timed out"}
        elif isinstance(e, requests.exceptions.HTTPError):
            status_code = getattr(e.response, "status_code", "unknown")
            return {"error": f"HTTP {status_code} error: {str(e)}"}
        elif isinstance(e, requests.exceptions.RequestException):
            return {"error": f"Request failed: {str(e)}"}
        elif isinstance(e, json.JSONDecodeError):
            return {"error": "Invalid JSON response from Databricks"}
        else:
            return {"error": f"Unexpected error: {str(e)}"}

    def _non_stream_response(
        self, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Handle non-streaming response."""
        payload["stream"] = False
        url = f"https://{self.valves.databricks_host.strip()}/serving-endpoints/{payload.get('model', '')}/invocations"

        try:
            response = self._make_request("POST", url, headers=headers, json=payload)
            return self._process_completion_response(response.json())
        except Exception as e:
            return self._handle_request_error(e)

    def _process_completion_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process completion response from Databricks."""
        if not isinstance(result, dict) or not (choices := result.get("choices", [])):
            return {
                "error": (
                    "Invalid response format"
                    if not isinstance(result, dict)
                    else "No response choices found"
                )
            }

        choice = choices[0]
        message = choice.get("message", {})
        if (content := message.get("content")) is None:
            return {"error": "No content in response"}

        return {
            "id": result.get("id", f"chatcmpl-{uuid.uuid4()}"),
            "object": "chat.completion",
            "created": result.get("created", int(time.time())),
            "model": result.get("model", "databricks-model"),
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": content,
                    },
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
            ],
            "usage": result.get("usage", {}),
        }
