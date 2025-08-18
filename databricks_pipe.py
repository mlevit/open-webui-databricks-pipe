"""
Title: Databricks LLM Pipe (Open WebUI)
"""

import requests
import json
import time
import uuid
import logging
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

    def __init__(self):
        self.name = ""
        self.valves = self.Valves()

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
        try:
            if not self.valves.custom_models.strip():
                logging.debug("No custom models configured (empty string)")
                return []

            logging.debug(
                f"Parsing custom models from configuration: {self.valves.custom_models.strip()}"
            )

            custom_models = json.loads(self.valves.custom_models.strip())
            logging.debug(f"Parsed JSON successfully, type: {type(custom_models)}")

            if not isinstance(custom_models, list):
                logging.warning(
                    f"Custom models configuration is not a list, got {type(custom_models)}"
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
                    logging.debug(f"Valid custom model {i}: {valid_model}")
                else:
                    invalid_count += 1
                    logging.warning(
                        f"Invalid custom model {i}: {model} (missing 'id' or 'name' keys)"
                    )

            if invalid_count > 0:
                logging.warning(f"Skipped {invalid_count} invalid custom model entries")

            logging.info(f"Successfully parsed {len(valid_models)} custom models")
            return valid_models

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse custom models JSON: {str(e)}")
            logging.error(f"Invalid JSON string: {self.valves.custom_models}")
            return []
        except Exception as e:
            logging.error(
                f"Unexpected error parsing custom models: {str(e)}", exc_info=True
            )
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
        try:
            # Check if auto-discovery is disabled
            if not self.valves.auto_discover_models:
                logging.info("Model auto-discovery is disabled in configuration")
                return []

            # Check configuration
            config_error = self._validate_config()
            if config_error:
                logging.warning(f"Configuration validation failed: {config_error}")
                return []

            logging.info(
                f"Starting model discovery for host: {self.valves.databricks_host.strip()}"
            )

            url = f"https://{self.valves.databricks_host.strip()}/api/2.0/serving-endpoints"
            headers = self._get_headers()

            logging.debug(f"Making request to: {url}")
            logging.debug(
                f"Request headers: {dict(headers, Authorization='Bearer ***')}"
            )  # Mask token

            response = requests.get(url, headers=headers)
            logging.info(f"API Response status: {response.status_code}")

            response.raise_for_status()

            response_data = response.json()
            all_endpoints = response_data.get("endpoints", [])
            logging.info(f"Found {len(all_endpoints)} total endpoints")

            if not all_endpoints:
                logging.warning("No endpoints found in API response")
                logging.debug(f"Full API response: {response_data}")
                return []

            # Filter endpoints step by step with logging
            named_endpoints = [ep for ep in all_endpoints if ep.get("name")]
            logging.info(f"Endpoints with names: {len(named_endpoints)}")

            ready_endpoints = [
                ep
                for ep in named_endpoints
                if ep.get("state", {}).get("ready") == "READY"
            ]
            logging.info(f"Ready endpoints: {len(ready_endpoints)}")

            # Log details about non-ready endpoints
            non_ready = [ep for ep in named_endpoints if ep not in ready_endpoints]
            if non_ready:
                logging.info("Non-ready endpoints:")
                for ep in non_ready:
                    state = ep.get("state", {})
                    logging.info(
                        f"  - {ep.get('name')}: ready={state.get('ready')}, config_update={state.get('config_update')}"
                    )

            llm_endpoints = [ep for ep in ready_endpoints if self._is_llm_endpoint(ep)]
            logging.info(f"LLM endpoints: {len(llm_endpoints)}")

            # Log details about filtered out endpoints
            non_llm = [ep for ep in ready_endpoints if ep not in llm_endpoints]
            if non_llm:
                logging.info("Non-LLM endpoints (filtered out):")
                for ep in non_llm:
                    endpoint_type = ep.get("endpoint_type", "N/A")
                    task = ep.get("task", "N/A")
                    logging.info(
                        f"  - {ep.get('name')}: endpoint_type={endpoint_type}, task={task}"
                    )

            discovered_models = [
                {"id": endpoint["name"], "name": endpoint["name"]}
                for endpoint in llm_endpoints
            ]

            logging.info(
                f"Successfully discovered {len(discovered_models)} LLM models:"
            )
            for model in discovered_models:
                logging.info(f"  - {model['name']}")

            return discovered_models

        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP request failed during model discovery: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                logging.error(f"Response status: {e.response.status_code}")
                logging.error(f"Response content: {e.response.text}")
            return []
        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to parse JSON response during model discovery: {str(e)}"
            )
            return []
        except Exception as e:
            logging.error(
                f"Unexpected error during model discovery: {str(e)}", exc_info=True
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
                    f"Using {len(custom_models)} custom models from configuration:"
                )
                for model in custom_models:
                    logging.info(f"  - {model['name']} (id: {model['id']})")
                return custom_models
            else:
                logging.info("No custom models configured")

            # Priority 2: Auto-discovered models
            logging.info("Attempting auto-discovery of models")
            discovered_models = self._discover_models()
            if discovered_models:
                logging.info(f"Using {len(discovered_models)} auto-discovered models")
                return discovered_models
            else:
                logging.warning("No models discovered through auto-discovery")

            # No models available
            logging.warning(
                "No models available from any source (custom or auto-discovery)"
            )
            return []

        except Exception as e:
            logging.error(f"Unexpected error in _get_models: {str(e)}", exc_info=True)
            return []

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

            logging.info(f"Returning {len(result)} models to Open WebUI")
            return result

        except Exception as e:
            logging.error(f"Error in pipes() method: {str(e)}", exc_info=True)
            return []

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
