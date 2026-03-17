# Copyright (c) 2025, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import logging

import requests

from datacustomcode.credentials import Credentials
from datacustomcode.proxy.client.base import BaseProxyClient

logger = logging.getLogger(__name__)

MODELS_API_BASE_URL = "https://api.salesforce.com"


class ModelsApiProxyClientProvider(BaseProxyClient):
    """Proxy client that calls the Salesforce Models API ``generateText`` endpoint."""

    CONFIG_NAME = "ModelsApiProxyClientProvider"

    def __init__(
        self,
        credentials_profile: str = "default",
        models_api_url: str = MODELS_API_BASE_URL,
        **kwargs: object,
    ) -> None:
        self._credentials_profile = credentials_profile
        self._models_api_url = models_api_url.rstrip("/")

    def call_llm_gateway(self, llmModelId: str, prompt: str, maxTokens: int) -> str:
        """Call the Models API ``generateText`` endpoint.

        Args:
            llmModelId: The model API name
                (e.g. ``"sfdc_ai__DefaultOpenAIGPT4OmniMini"``).
            prompt: The text prompt to send.
            maxTokens: Maximum number of tokens to generate.

        Returns:
            The generated text string from the API response.

        Raises:
            RuntimeError: On HTTP errors or unexpected response shapes.
        """
        access_token = self._get_access_token()

        url = (
            f"{self._models_api_url}"
            f"/einstein/platform/v1/models/{llmModelId}/generations"
        )
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "x-sfdc-app-context": "EinsteinGPT",
            "x-client-feature-id": "ai-platform-models-connected-app",
        }
        body = {"prompt": prompt, "maxTokens": maxTokens}

        logger.debug(f"POST {url} (maxTokens={maxTokens})")

        try:
            response = requests.post(url, json=body, headers=headers, timeout=120)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Models API request failed: {exc}"
            ) from exc

        if response.status_code >= 300:
            error_msg = response.text
            try:
                error_data = response.json()
                if isinstance(error_data, dict) and "error" in error_data:
                    error_msg = error_data["error"]
                elif isinstance(error_data, list) and error_data:
                    error_msg = error_data[0].get("message", error_msg)
            except (json.JSONDecodeError, KeyError):
                pass
            raise RuntimeError(
                f"Models API request failed (HTTP {response.status_code}): {error_msg}"
            )

        try:
            result = response.json()
            return result["generation"]["generatedText"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected response from Models API: {response.text}"
            ) from exc

    def _get_access_token(self) -> str:
        """Obtain an access token via a client-credentials exchange.

        The Models API only supports the client-credentials grant type.
        """
        credentials = Credentials.from_available(profile=self._credentials_profile)

        token_url = f"{credentials.login_url}/services/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
        }

        try:
            response = requests.post(token_url, data=payload, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Client credentials token request failed: {exc}"
            ) from exc

        if response.status_code >= 300:
            raise RuntimeError(
                f"Client credentials token request failed "
                f"(HTTP {response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse token response: {response.text}"
            ) from exc

        access_token = data.get("access_token")
        if not access_token:
            raise RuntimeError("Token response missing access_token")

        return access_token
