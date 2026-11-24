# ============================================================
# DatabricksMastery
# GenAI / 01-LLM-Fundamentals
# 01_Foundation_Model_Inference
#
# Experiment:
# Foundation Model Inference, Response Inspection,
# and MLflow Tracking
# ============================================================


# ============================================================
# 1. INSTALL DEPENDENCIES
# ============================================================

#%pip install -U openai


# ============================================================
# 2. IMPORTS
# ============================================================

import json
import mlflow

from openai import OpenAI


# ============================================================
# 3. CONFIGURE DATABRICKS AI GATEWAY
# ============================================================

workspace_url = (
    "https://" + spark.conf.get("spark.databricks.workspaceUrl")
)

MODEL_ENDPOINT = "system.ai.qwen35-122b-a10b"

token = (
    dbutils.notebook.entry_point
    .getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .get()
)

client = OpenAI(
    api_key=token,
    base_url=f"{workspace_url}/ai-gateway/mlflow/v1"
)

print("Workspace:", workspace_url)
print("Model endpoint:", MODEL_ENDPOINT)
print("Client initialized successfully.")


# ============================================================
# 4. DEFINE INPUT MESSAGES
# ============================================================

messages = [
    {
        "role": "system",
        "content": (
            "You are a senior software architect. "
            "Provide clear, technically accurate explanations."
        )
    },
    {
        "role": "user",
        "content": (
            "Explain the difference between traditional machine "
            "learning models and Large Language Models. "
            "Use three concise technical points."
        )
    }
]


# ============================================================
# 5. CONFIGURE INFERENCE PARAMETERS
# ============================================================

inference_config = {
    "model": MODEL_ENDPOINT,
    "temperature": 0.2,
    "max_tokens": 300,
    "top_p": 1.0
}

print("Inference configuration:")
print(json.dumps(inference_config, indent=2))


# ============================================================
# 6. RUN MODEL INFERENCE
# ============================================================

response = client.chat.completions.create(
    model=MODEL_ENDPOINT,
    messages=messages,
    temperature=inference_config["temperature"],
    max_tokens=1500,
    top_p=inference_config["top_p"]
)

# ============================================================
# RESPONSE DIAGNOSTICS
# ============================================================

message = response.choices[0].message

print("Finish reason:", response.choices[0].finish_reason)
print("Content:", repr(message.content))

if response.usage:
    print("Prompt tokens:", response.usage.prompt_tokens)
    print("Completion tokens:", response.usage.completion_tokens)
    print("Total tokens:", response.usage.total_tokens)

content = message.content

print("\n" + "=" * 70)
print("QWEN RESPONSE DIAGNOSTICS")
print("=" * 70)

print("Response ID:", response.id)
print("Model:", response.model)
print("Finish reason:", response.choices[0].finish_reason)
print("Message type:", type(message).__name__)
print("Content type:", type(content).__name__)

print("\nRaw content:")
print(repr(content))


def inspect_structure(value, path="content", depth=0):

    indent = "  " * depth

    if depth > 8:
        return

    if isinstance(value, dict):

        print(f"{indent}{path}: dict")
        print(f"{indent}Keys: {list(value.keys())}")

        if value.get("type") == "reasoning":
            print(f"{indent}Reasoning block skipped.")
            return

        for key, item in value.items():
            inspect_structure(item, f"{path}.{key}", depth + 1)

    elif isinstance(value, list):

        print(f"{indent}{path}: list ({len(value)} items)")

        for index, item in enumerate(value):
            inspect_structure(item, f"{path}[{index}]", depth + 1)

    elif isinstance(value, str):

        print(f"{indent}{path}: string")

        if any(
            key in path.lower()
            for key in ("text", "output_text", "content")
        ):
            print(f"{indent}Value: {value[:2000]}")

    else:
        print(f"{indent}{path}: {type(value).__name__}")


print("\nNested structure:")
inspect_structure(content)

print("\n" + "=" * 70)

# ============================================================
# 7. INSPECT RESPONSE STRUCTURE
# ============================================================

def inspect_response_content(content):
    """
    Inspect the response structure without exposing
    reasoning content.
    """

    print("Content type:", type(content).__name__)

    if isinstance(content, list):

        print("\nStructured blocks:")

        for index, block in enumerate(content):

            if isinstance(block, dict):
                block_type = block.get("type")
                keys = list(block.keys())
                block_text = block.get("text")

            else:
                block_type = getattr(block, "type", None)
                keys = (
                    list(block.keys())
                    if hasattr(block, "keys")
                    else None
                )
                block_text = getattr(block, "text", None)

            print(f"\nBlock {index}")
            print("Type:", block_type)
            print("Available keys:", keys)

            # Display only user-facing text blocks.
            if block_type in ("text", "output_text"):
                print("Text:", repr(block_text))

    else:
        print("Raw content:", repr(content))


# ============================================================
# 8. EXTRACT USER-FACING RESPONSE TEXT
# ============================================================

def extract_response_text(content):
    """
    Extract user-facing text from the model response.

    Reasoning and other non-text blocks are intentionally ignored.
    """

    if isinstance(content, str):
        return content.strip()

    if content is None:
        return ""

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):
                block_type = block.get("type")
                block_text = block.get("text")

            else:
                block_type = getattr(block, "type", None)
                block_text = getattr(block, "text", None)

            if (
                block_type in ("text", "output_text")
                and isinstance(block_text, str)
            ):
                text_parts.append(block_text)

        return "\n".join(text_parts).strip()

    return ""


# Inspect the response.
inspect_response_content(message.content)

# Extract user-facing text.
answer = extract_response_text(message.content)

if not answer:

    print("\nNo user-facing text could be extracted.")
    print("Raw content representation:")
    print(repr(message.content))

    raise ValueError(
        "Could not extract user-facing text. "
        "Inspect the response structure above before proceeding."
    )

print("\nExtracted answer:")
print("=" * 70)
print(answer)
print("=" * 70)


# ============================================================
# 9. INSPECT RESPONSE METADATA
# ============================================================

usage = response.usage

response_metadata = {
    "response_id": response.id,
    "model": response.model,
    "finish_reason": response.choices[0].finish_reason,
    "content_type": type(message.content).__name__,
    "usage": {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": (
            usage.completion_tokens if usage else None
        ),
        "total_tokens": usage.total_tokens if usage else None
    }
}

print("\nResponse metadata:")
print(json.dumps(response_metadata, indent=2))


# ============================================================
# 10. CONFIGURE MLFLOW EXPERIMENT
# ============================================================

current_user = spark.sql(
    "SELECT current_user()"
).first()[0]

experiment_path = (
    f"/Users/{current_user}/DatabricksExperiments/"
    "GenAI/01-LLM-Fundamentals"
)

mlflow.set_experiment(experiment_path)

print("\nMLflow experiment:", experiment_path)


# ============================================================
# 11. TRACK INFERENCE WITH MLFLOW
# ============================================================

with mlflow.start_run(
    run_name="qwen_foundation_model_inference"
):

    # Log model and inference parameters.
    mlflow.log_params({
        "model": MODEL_ENDPOINT,
        "temperature": inference_config["temperature"],
        "max_tokens": inference_config["max_tokens"],
        "top_p": inference_config["top_p"]
    })

    # Log token usage.
    if usage:

        mlflow.log_metrics({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        })

    # Log input messages.
    mlflow.log_text(
        json.dumps(messages, indent=2),
        "input_messages.json"
    )

    # Log generated output.
    mlflow.log_text(
        answer,
        "generated_response.txt"
    )

    # Log response metadata.
    mlflow.log_text(
        json.dumps(response_metadata, indent=2),
        "response_metadata.json"
    )

    run_id = mlflow.active_run().info.run_id

    print("\nMLflow tracking completed.")
    print("Run ID:", run_id)


# ============================================================
# 12. FINAL EXPERIMENT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

print("Model:", MODEL_ENDPOINT)
print("Temperature:", inference_config["temperature"])
print("Maximum tokens:", inference_config["max_tokens"])
print("Top P:", inference_config["top_p"])

if usage:
    print("Prompt tokens:", usage.prompt_tokens)
    print("Completion tokens:", usage.completion_tokens)
    print("Total tokens:", usage.total_tokens)

print("\nGenerated answer:")
print(answer)

print("\nMLflow experiment:", experiment_path)
print("MLflow run ID:", run_id)

print("\nExperiment completed successfully.")
