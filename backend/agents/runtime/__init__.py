"""Provider-agnostic agent runtime API."""

from backend.agents.runtime.base import (
    AgentRuntime,
    AgentRuntimeSpec,
    ProviderConfigurationError,
    ProviderFeatureUnsupported,
    ProviderName,
    StreamEvent,
    StreamText,
    StreamToolCall,
    StreamUsage,
    ToolSpec,
)
from backend.agents.runtime.factory import (
    make_runtime,
    selected_provider,
    validate_provider_credentials,
)

__all__ = [
    "AgentRuntime",
    "AgentRuntimeSpec",
    "ProviderConfigurationError",
    "ProviderFeatureUnsupported",
    "ProviderName",
    "StreamEvent",
    "StreamText",
    "StreamToolCall",
    "StreamUsage",
    "ToolSpec",
    "make_runtime",
    "selected_provider",
    "validate_provider_credentials",
]
