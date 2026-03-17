from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests as req_lib

from datacustomcode.credentials import Credentials
from datacustomcode.proxy.client.ModelsApiProxyClientProvider import (
    MODELS_API_BASE_URL,
    ModelsApiProxyClientProvider,
)

ACCESS_TOKEN = "test-access-token"

def _mock_response(
    status_code: int = 200,
    json_body: dict | None = None,
    text: str = "",
) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.json.return_value = json_body or {}
    return response


def _generation_response(generated_text: str = "Hello world") -> dict:
    return {
        "generation": {
            "generatedText": generated_text,
            "contentQuality": {},
            "parameters": {},
        }
    }


def _mock_credentials():
    return Credentials(
        login_url="https://login.salesforce.com",
        client_id="cid",
        client_secret="csecret",
        auth_type="client_credentials",
    )


def _patch_credentials(creds):
    return patch.object(Credentials, "from_available", return_value=creds)


def _patch_post(return_value=None, side_effect=None):
    kwargs = {}
    if return_value is not None:
        kwargs["return_value"] = return_value
    if side_effect is not None:
        kwargs["side_effect"] = side_effect
    return patch(
        "datacustomcode.proxy.client.ModelsApiProxyClientProvider.requests.post",
        **kwargs,
    )

class TestConfigName:
    def test_config_name_is_set(self):
        assert ModelsApiProxyClientProvider.CONFIG_NAME == "ModelsApiProxyClientProvider"

class TestCallLlmGateway:
    @pytest.fixture
    def provider(self):
        return ModelsApiProxyClientProvider(credentials_profile="myprofile")

    @pytest.fixture
    def creds(self):
        return _mock_credentials()

    def test_posts_to_correct_url(self, provider, creds):
        with _patch_credentials(creds), _patch_post(
            return_value=_mock_response(
                json_body={"access_token": "tok"}
            )
        ):
            # First call is token exchange, second is the API call.
            # Patch post to return token on first call, generation on second.
            pass

        # Use a simpler approach: patch _get_access_token directly
        with patch.object(provider, "_get_access_token", return_value=ACCESS_TOKEN):
            with _patch_post(
                return_value=_mock_response(json_body=_generation_response())
            ) as mock_post:
                provider.call_llm_gateway("myModel", "hello", 100)

        url = mock_post.call_args[0][0]
        assert url == (
            f"{MODELS_API_BASE_URL}"
            f"/einstein/platform/v1/models/myModel/generations"
        )

    def test_custom_models_api_url(self):
        provider = ModelsApiProxyClientProvider(
            models_api_url="https://test.api.salesforce.com",
        )
        with patch.object(provider, "_get_access_token", return_value=ACCESS_TOKEN):
            with _patch_post(
                return_value=_mock_response(json_body=_generation_response())
            ) as mock_post:
                provider.call_llm_gateway("myModel", "hello", 100)

        url = mock_post.call_args[0][0]
        assert url == (
            "https://test.api.salesforce.com"
            "/einstein/platform/v1/models/myModel/generations"
        )

    def test_sends_correct_headers(self, provider):
        with patch.object(provider, "_get_access_token", return_value=ACCESS_TOKEN):
            with _patch_post(
                return_value=_mock_response(json_body=_generation_response())
            ) as mock_post:
                provider.call_llm_gateway("myModel", "hello", 100)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {ACCESS_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["x-sfdc-app-context"] == "EinsteinGPT"
        assert headers["x-client-feature-id"] == "ai-platform-models-connected-app"

    def test_sends_correct_body(self, provider):
        with patch.object(provider, "_get_access_token", return_value=ACCESS_TOKEN):
            with _patch_post(
                return_value=_mock_response(json_body=_generation_response())
            ) as mock_post:
                provider.call_llm_gateway("myModel", "test prompt", 200)

        body = mock_post.call_args.kwargs["json"]
        assert body == {"prompt": "test prompt", "maxTokens": 200}

    def test_returns_generated_text(self, provider):
        with patch.object(provider, "_get_access_token", return_value=ACCESS_TOKEN):
            with _patch_post(
                return_value=_mock_response(
                    json_body=_generation_response("LLM says hi")
                )
            ):
                result = provider.call_llm_gateway("myModel", "hello", 100)

        assert result == "LLM says hi"

    def test_http_error_raises_runtime_error(self, provider):
        with patch.object(provider, "_get_access_token", return_value=ACCESS_TOKEN):
            with _patch_post(
                return_value=_mock_response(status_code=401, text="Unauthorized")
            ):
                with pytest.raises(RuntimeError, match="HTTP 401"):
                    provider.call_llm_gateway("myModel", "hello", 100)

    def test_http_error_extracts_error_field(self, provider):
        error_body = {"error": "Model not found"}
        with patch.object(provider, "_get_access_token", return_value=ACCESS_TOKEN):
            with _patch_post(
                return_value=_mock_response(
                    status_code=404, json_body=error_body, text="Not Found"
                )
            ):
                with pytest.raises(RuntimeError, match="Model not found"):
                    provider.call_llm_gateway("myModel", "hello", 100)

    def test_request_exception_raises_runtime_error(self, provider):
        with patch.object(provider, "_get_access_token", return_value=ACCESS_TOKEN):
            with _patch_post(
                side_effect=req_lib.RequestException("connection refused")
            ):
                with pytest.raises(RuntimeError, match="Models API request failed"):
                    provider.call_llm_gateway("myModel", "hello", 100)

    def test_invalid_response_json_raises_runtime_error(self, provider):
        bad_response = _mock_response(status_code=200, text="not json")
        bad_response.json.side_effect = json.JSONDecodeError("bad json", "", 0)
        with patch.object(provider, "_get_access_token", return_value=ACCESS_TOKEN):
            with _patch_post(return_value=bad_response):
                with pytest.raises(RuntimeError, match="Unexpected response"):
                    provider.call_llm_gateway("myModel", "hello", 100)

    def test_missing_generation_key_raises_runtime_error(self, provider):
        with patch.object(provider, "_get_access_token", return_value=ACCESS_TOKEN):
            with _patch_post(
                return_value=_mock_response(json_body={"other": "data"})
            ):
                with pytest.raises(RuntimeError, match="Unexpected response"):
                    provider.call_llm_gateway("myModel", "hello", 100)

class TestGetAccessToken:
    @pytest.fixture
    def provider(self):
        return ModelsApiProxyClientProvider(credentials_profile="myprofile")

    @pytest.fixture
    def creds(self):
        return _mock_credentials()

    def test_returns_token(self, provider, creds):
        token_response = _mock_response(
            json_body={"access_token": "cc-token"}
        )
        with _patch_credentials(creds), _patch_post(
            return_value=token_response
        ) as mock_post:
            token = provider._get_access_token()

        assert token == "cc-token"
        call_args = mock_post.call_args
        assert (
            call_args[0][0]
            == "https://login.salesforce.com/services/oauth2/token"
        )
        assert call_args.kwargs["data"]["grant_type"] == "client_credentials"
        assert call_args.kwargs["data"]["client_id"] == "cid"
        assert call_args.kwargs["data"]["client_secret"] == "csecret"

    def test_http_error_raises(self, provider, creds):
        with _patch_credentials(creds), _patch_post(
            return_value=_mock_response(status_code=401, text="invalid_client")
        ):
            with pytest.raises(
                RuntimeError, match="Client credentials token request failed"
            ):
                provider._get_access_token()

    def test_request_exception_raises(self, provider, creds):
        with _patch_credentials(creds), _patch_post(
            side_effect=req_lib.RequestException("network error")
        ):
            with pytest.raises(
                RuntimeError, match="Client credentials token request failed"
            ):
                provider._get_access_token()

    def test_missing_access_token_raises(self, provider, creds):
        token_response = _mock_response(
            json_body={"instance_url": "https://x.salesforce.com"}
        )
        with _patch_credentials(creds), _patch_post(return_value=token_response):
            with pytest.raises(RuntimeError, match="missing access_token"):
                provider._get_access_token()



