"""
Title: Databricks LLM Pipe (Open WebUI)
"""

from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Union, Generator, Dict, Any, Optional, Literal
import json
import logging
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
            description="Databricks host name (required)",
        )

        # Authentication configuration
        authentication_method: Literal["Personal Access Token", "OAuth"] = Field(
            default="Personal Access Token",
            description="Authentication method",
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

    def _validate_config(self) -> Union[str, None]:
        """Validate configuration. Returns error message or None if valid."""
        if not self.valves.databricks_host.strip():
            return "Databricks host is required."

        # Validate authentication method and required fields
        if self.valves.authentication_method.strip() == AuthType.PAT.value:
            if not self.valves.personal_access_token.strip():
                return "Personal Access Token authentication selected: Please configure your Personal Access Token."
        elif self.valves.authentication_method.strip() == AuthType.OAUTH.value:
            if (
                not self.valves.oauth_client_id.strip()
                or not self.valves.oauth_secret.strip()
            ):
                return "OAuth authentication selected: Please configure oauth_client_id and oauth_secret."
        else:
            return f"Invalid authentication_method: {self.valves.authentication_method}. Must be 'Personal Access Token' or 'OAuth'."

        return None

    def _is_token_expired(self) -> bool:
        """Check if the current OAuth token is expired or will expire soon."""
        if not self._token_expires_at:
            return True

        # Consider token expired if it expires within the next 5 minutes
        buffer_time = timedelta(minutes=5)
        return datetime.utcnow() + buffer_time >= self._token_expires_at

    def _request_token(self, data: Dict[str, str]) -> Union[str, None]:
        """Make OAuth token request and process response."""
        try:
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
        except Exception as e:
            logging.error("OAuth token request failed: %s", str(e))
            return None

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

        token = self._request_token(data)
        if not token:
            # Clear tokens and get new one
            self._access_token = None
            self._refresh_token = None
            self._token_expires_at = None
            return self._get_oauth_token()
        return token

    def _get_headers(self) -> Union[Dict[str, str], None]:
        """Generate request headers based on configured authentication method."""
        # Select token based on configured authentication method
        if self.valves.authentication_method.strip() == AuthType.PAT.value:
            token = self.valves.personal_access_token.strip()
        else:  # OAuth
            token = self._get_oauth_token()

        return (
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
            if token
            else None
        )

    def _parse_custom_models(self) -> List[Dict[str, str]]:
        """Parse custom models from valves configuration."""
        try:
            if not self.valves.custom_models.strip():
                logging.debug("No custom models configured (empty string)")
                return []

            logging.debug(
                "Parsing custom models from configuration: %s",
                self.valves.custom_models.strip(),
            )

            custom_models = json.loads(self.valves.custom_models.strip())
            logging.debug("Parsed JSON successfully, type: %s", type(custom_models))

            if not isinstance(custom_models, list):
                logging.warning(
                    "Custom models configuration is not a list, got %s",
                    type(custom_models),
                )
                return []

            valid_models = []
            invalid_count = 0

            for i, model in enumerate(custom_models):
                if isinstance(model, dict) and "id" in model and "name" in model:
                    valid_model = {
                        "id": str(model["id"]).strip(),
                        "name": str(model["name"]).strip(),
                    }
                    valid_models.append(valid_model)
                    logging.debug("Valid custom model %s: %s", i, valid_model)
                else:
                    invalid_count += 1
                    logging.warning(
                        "Invalid custom model %s: %s (missing 'id' or 'name' keys)",
                        i,
                        model,
                    )

            if invalid_count > 0:
                logging.warning(
                    "Skipped %s invalid custom model entries", invalid_count
                )

            logging.info("Successfully parsed %s custom models", len(valid_models))
            return valid_models

        except json.JSONDecodeError as e:
            logging.error("Failed to parse custom models JSON: %s", str(e))
            logging.error("Invalid JSON string: %s", self.valves.custom_models)
            return []
        except Exception as e:
            logging.error(
                "Unexpected error parsing custom models: %s", str(e), exc_info=True
            )
            return []

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
        try:
            # Check if auto-discovery is disabled
            if not self.valves.auto_discover_models:
                logging.info("Model auto-discovery is disabled in configuration")
                return []

            # Check configuration
            config_error = self._validate_config()
            if config_error:
                logging.warning("Configuration validation failed: %s", config_error)
                return []

            logging.info(
                "Starting model discovery for host: %s",
                self.valves.databricks_host.strip(),
            )

            url = f"https://{self.valves.databricks_host.strip()}/api/2.0/serving-endpoints"
            headers = self._get_headers()

            if not headers:
                logging.error(
                    "Failed to get authentication headers for model discovery"
                )
                return []

            logging.debug("Making request to: %s", url)
            logging.debug(
                "Request headers: %s", dict(headers, Authorization="Bearer ***")
            )  # Mask token

            response = requests.get(url, headers=headers)
            logging.info("API Response status: %s", response.status_code)

            response.raise_for_status()

            response_data = response.json()
            all_endpoints = response_data.get("endpoints", [])
            logging.info("Found %s total endpoints", len(all_endpoints))

            if not all_endpoints:
                logging.warning("No endpoints found in API response")
                logging.debug("Full API response: %s", response_data)
                return []

            # Filter endpoints step by step with logging
            named_endpoints = [ep for ep in all_endpoints if ep.get("name")]
            logging.info("Endpoints with names: %s", len(named_endpoints))

            ready_endpoints = [
                ep
                for ep in named_endpoints
                if ep.get("state", {}).get("ready") == "READY"
            ]
            logging.info("Ready endpoints: %s", len(ready_endpoints))

            # Log details about non-ready endpoints
            non_ready = [ep for ep in named_endpoints if ep not in ready_endpoints]
            if non_ready:
                logging.info("Non-ready endpoints:")
                for ep in non_ready:
                    state = ep.get("state", {})
                    logging.info(
                        "  - %s: ready=%s, config_update=%s",
                        ep.get("name"),
                        state.get("ready"),
                        state.get("config_update"),
                    )

            llm_endpoints = [ep for ep in ready_endpoints if self._is_llm_endpoint(ep)]
            logging.info("LLM endpoints: %s", len(llm_endpoints))

            # Log details about filtered out endpoints
            non_llm = [ep for ep in ready_endpoints if ep not in llm_endpoints]
            if non_llm:
                logging.info("Non-LLM endpoints (filtered out):")
                for ep in non_llm:
                    endpoint_type = ep.get("endpoint_type", "N/A")
                    task = ep.get("task", "N/A")
                    logging.info(
                        "  - %s: endpoint_type=%s, task=%s",
                        ep.get("name"),
                        endpoint_type,
                        task,
                    )

            discovered_models = [
                {"id": endpoint["name"], "name": endpoint["name"]}
                for endpoint in llm_endpoints
            ]

            logging.info(
                "Successfully discovered %s LLM models:", len(discovered_models)
            )
            for model in discovered_models:
                logging.info("  - %s", model["name"])

            return discovered_models

        except requests.exceptions.RequestException as e:
            logging.error("HTTP request failed during model discovery: %s", str(e))
            if hasattr(e, "response") and e.response is not None:
                logging.error("Response status: %s", e.response.status_code)
                logging.error("Response content: %s", e.response.text)
            return []
        except json.JSONDecodeError as e:
            logging.error(
                "Failed to parse JSON response during model discovery: %s", str(e)
            )
            return []
        except Exception as e:
            logging.error(
                "Unexpected error during model discovery: %s", str(e), exc_info=True
            )
            return []

    def _get_models(self) -> List[Dict[str, str]]:
        """Get models using priority: custom -> auto-discovery."""
        try:
            logging.info("Starting model retrieval process")

            # Priority 1: Custom models from configuration
            logging.debug("Checking for custom models from configuration")
            custom_models = self._parse_custom_models()
            if custom_models:
                logging.info(
                    "Using %s custom models from configuration:", len(custom_models)
                )
                for model in custom_models:
                    logging.info("  - %s (id: %s)", model["name"], model["id"])
                return custom_models
            else:
                logging.info("No custom models configured")

            # Priority 2: Auto-discovered models
            logging.info("Attempting auto-discovery of models")
            discovered_models = self._discover_models()
            if discovered_models:
                logging.info("Using %s auto-discovered models", len(discovered_models))
                return discovered_models
            else:
                logging.warning("No models discovered through auto-discovery")

            # No models available
            logging.warning(
                "No models available from any source (custom or auto-discovery)"
            )
            return []

        except Exception as e:
            logging.error("Unexpected error in _get_models: %s", str(e), exc_info=True)
            return []

    def pipes(self) -> List[Dict[str, Any]]:
        """Return available model configurations."""
        try:
            logging.info("Open WebUI requesting available models")
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

            logging.info("Returning %s models to Open WebUI", len(result))
            return result

        except Exception as e:
            logging.error("Error in pipes() method: %s", str(e), exc_info=True)
            return []

    def pipe(
        self, body: Dict[str, Any], __user__: Dict[str, Any] = None
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Main pipe method to handle both streaming and non-streaming requests.

        Args:
            body: Request body containing messages and parameters
            __user__: User information from Open WebUI (optional)

        Returns:
            Response dictionary or generator for streaming responses
        """
        # Validate configuration
        config_error = self._validate_config()
        if config_error:
            return {"error": f"Configuration error: {config_error}"}

        # Prepare payload
        payload = self._prepare_payload(body, __user__)
        headers = self._get_headers()

        if not headers:
            return {"error": "Authentication failed: could not get valid headers"}

        try:
            if body.get("stream", False):
                return self._stream_response(payload, headers)
            else:
                return self._non_stream_response(payload, headers)
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

    def _prepare_payload(
        self, body: Dict[str, Any], user: Dict[str, Any] = None
    ) -> Dict[str, Any]:
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

        # Add usage_context with user email if user information is available
        if user and user.get("email"):
            payload["usage_context"] = {"user_email": user.get("email")}

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
