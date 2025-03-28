import os
from litellm import completion
os.environ["OPENROUTER_API_KEY"] = "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"

#os.environ["OR_SITE_URL"] = "" # optional
#os.environ["OR_APP_NAME"] = "" # optional

messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is the capital of Spain?"}
]


response = completion(
            model="openrouter/cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            messages=messages,
        )

print(response)