"""
Title: Databricks LLM Pipe (Open WebUI)
"""

from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Union, Generator, Dict, Any, Optional, Literal
import json
import requests
import time
import uuid


class AuthType(Enum):
    """Supported authentication types for Databricks."""

    PAT = "Personal Access Token"
    OAUTH = "OAuth"


class Pipe:
    """
    Databricks LLM integration pipe optimized for Open WebUI.
    """

    class Valves(BaseModel):
        """Configuration model with validation and sensible defaults."""

        databricks_host: str = Field(
            default="your-workspace.cloud.databricks.com",
            description="Databricks host name",
        )

        # Authentication configuration
        authentication_method: Literal["Personal Access Token", "OAuth"] = Field(
            default="Personal Access Token",
            description="",
        )

        # === PAT Authentication Fields (only required when authentication_method = Personal Access Token) ===
        personal_access_token: str = Field(
            default="",
            description="",
        )

        # === OAuth Authentication Fields (only required when authentication_method = OAuth) ===
        oauth_client_id: str = Field(
            default="",
            description="",
        )
        oauth_secret: str = Field(
            default="",
            description="",
        )

        # === Model Configuration ===
        auto_discover_models: bool = Field(
            default=True,
            description="Automatically discover available LLM serving endpoints",
        )
        custom_models: str = Field(
            default="",
            description="Custom models (JSON format: [{'id': 'endpoint-name', 'name': 'Display Name'}]). Leave empty to use auto-discovery.",
        )

    # No constants needed for simplified approach

    def __init__(self):
        self.name = ""
        self.valves = self.Valves()
        # OAuth token management
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._refresh_token: Optional[str] = None

    def _is_token_expired(self) -> bool:
        """Check if the current OAuth token is expired or will expire soon."""
        if not self._token_expires_at:
            return True

        # Consider token expired if it expires within the next 5 minutes
        buffer_time = timedelta(minutes=5)
        return datetime.utcnow() + buffer_time >= self._token_expires_at

    def _request_token(self, data: Dict[str, str]) -> Union[str, None]:
        """Make OAuth token request and process response."""
        url = f"https://{self.valves.databricks_host.strip()}/oidc/v1/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()

        token_data = response.json()
        self._access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        if refresh_token := token_data.get("refresh_token"):
            self._refresh_token = refresh_token

        return self._access_token

    def _get_oauth_token(self) -> Union[str, None]:
        """Get OAuth access token for service principal."""
        if self._access_token and not self._is_token_expired():
            return self._access_token

        data = {
            "grant_type": "client_credentials",
            "scope": "all-apis",
            "client_id": self.valves.oauth_client_id.strip(),
            "client_secret": self.valves.oauth_secret.strip(),
        }
        return self._request_token(data)

    def _refresh_oauth_token(self) -> Union[str, None]:
        """Refresh OAuth token using refresh token."""
        if not self._refresh_token:
            return self._get_oauth_token()

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self.valves.oauth_client_id.strip(),
            "client_secret": self.valves.oauth_secret.strip(),
        }

        return self._request_token(data)

    def _get_headers(self) -> Dict[str, str]:
        """Generate request headers based on configured authentication method."""
        # Select token based on configured authentication method
        if self.valves.authentication_method.strip() == AuthType.PAT.value:
            token = self.valves.personal_access_token.strip()
        else:  # OAuth
            token = self._get_oauth_token()

        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def _parse_custom_models(self) -> List[Dict[str, str]]:
        """Parse custom models from valves configuration."""
        if not self.valves.custom_models.strip():
            return []

        custom_models = json.loads(self.valves.custom_models.strip())

        if not isinstance(custom_models, list):
            return []

        valid_models = []
        for model in custom_models:
            if isinstance(model, dict) and "id" in model and "name" in model:
                valid_model = {
                    "id": str(model["id"]).strip(),
                    "name": str(model["name"]).strip(),
                }
                valid_models.append(valid_model)

        return valid_models

    def _is_llm_endpoint(self, endpoint: Dict[str, Any]) -> bool:
        """Check if an endpoint matches the allowed LLM endpoint paths."""
        allowed_paths = [
            "agent/v1/chat",
            "agent/v1/responses",
            "agent/v2/chat",
            "llm/v1/chat",
            "llm/v1/completions",
        ]
        task = endpoint.get("task", "").lower()
        return task in allowed_paths

    def _discover_models(self) -> List[Dict[str, str]]:
        """Auto-discover available LLM serving endpoints from Databricks."""
        # Check if auto-discovery is disabled
        if not self.valves.auto_discover_models:
            return []

        url = f"https://{self.valves.databricks_host.strip()}/api/2.0/serving-endpoints"
        headers = self._get_headers()

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        response_data = response.json()
        all_endpoints = response_data.get("endpoints", [])

        # Filter endpoints
        named_endpoints = [ep for ep in all_endpoints if ep.get("name")]
        ready_endpoints = [
            ep for ep in named_endpoints if ep.get("state", {}).get("ready") == "READY"
        ]
        llm_endpoints = [ep for ep in ready_endpoints if self._is_llm_endpoint(ep)]

        discovered_models = [
            {"id": endpoint["name"], "name": endpoint["name"]}
            for endpoint in llm_endpoints
        ]

        return discovered_models

    def _get_models(self) -> List[Dict[str, str]]:
        """Get models using priority: custom -> auto-discovery."""
        # Priority 1: Custom models from configuration
        custom_models = self._parse_custom_models()
        if custom_models:
            return custom_models

        # Priority 2: Auto-discovered models
        discovered_models = self._discover_models()
        return discovered_models

    def pipes(self) -> List[Dict[str, Any]]:
        """Return available model configurations."""
        current_time = int(time.time())
        models = self._get_models()

        result = [
            {
                "id": model["id"],
                "name": model["name"],
                "object": "model",
                "created": current_time,
                "owned_by": "Databricks",
            }
            for model in models
        ]

        return result

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

        # Prepare payload
        payload = self._prepare_payload(body)
        headers = self._get_headers()

        if body.get("stream", False):
            return self._stream_response(payload, headers)
        else:
            return self._non_stream_response(payload, headers)

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

                chunk = json.loads(data_content)
                processed_chunk = self._process_stream_chunk(chunk)
                if processed_chunk:
                    yield f"data: {json.dumps(processed_chunk)}\n\n"

    def _process_stream_chunk(
        self, chunk: Dict[str, Any]
    ) -> Union[Dict[str, Any], None]:
        """Process individual streaming chunk."""
        choices = chunk.get("choices", [])
        choice = choices[0]
        delta = choice.get("delta", {})

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

    def _non_stream_response(
        self, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Handle non-streaming response."""
        payload["stream"] = False
        url = f"https://{self.valves.databricks_host.strip()}/serving-endpoints/{payload.get('model', '')}/invocations"

        response = self._make_request("POST", url, headers=headers, json=payload)
        return self._process_completion_response(response.json())

    def _process_completion_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process completion response from Databricks."""
        choices = result.get("choices", [])
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content")

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
