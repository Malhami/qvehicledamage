# Azure OpenAI service credentials
# endpoint = "https://mutas-m2yh5fm6-eastus2.openai.azure.com/"
# deployment = "gpt-4"
# subscription_key = "BxWLJ2EaFW6oY1L5I6WQs9SctYMXMLwDLSvPLXGxU7tCVkpHRYuFJQQJ99AKACHYHv6XJ3w3AAAAACOG0aOK"
# api_version ="2023-07-01-preview"

endpoint = "https://mutas-m2yh5fm6-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
deployment = "gpt-4o"
subscription_key = "BxWLJ2EaFW6oY1L5I6WQs9SctYMXMLwDLSvPLXGxU7tCVkpHRYuFJQQJ99AKACHYHv6XJ3w3AAAAACOG0aOK"
api_version ="2023-07-01-preview"
import os

llm_config = {
    "config_list": [
        {
            "model": deployment,  # or "gpt-3.5-turbo"
            "api_key": subscription_key,
            "base_url": endpoint,
            "api_type": "azure",
            "api_version": "2023-07-01-preview",

        }
    ],
    "temperature": 0.7,
}