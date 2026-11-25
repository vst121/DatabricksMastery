
# 07_Agentic_AI.py
# Build a controlled tool-calling agent using Databricks AI Gateway.

import json
from openai import OpenAI

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

MODEL_ENDPOINT = "system.ai.qwen35-122b-a10b"

workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

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

# ---------------------------------------------------------
# 2. Define application tools
# ---------------------------------------------------------

def calculate(expression: str) -> dict:
    """Evaluate a basic arithmetic expression safely."""

    import ast
    import operator

    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent too large.")

            return operations[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.operand))

        raise ValueError("Unsupported mathematical expression.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)

        if abs(result) > 1e12:
            raise ValueError("Result exceeds the allowed limit.")

        return {"result": result}

    except Exception as exc:
        return {"error": str(exc)}


def search_knowledge_base(query: str) -> dict:
    """Search a small in-memory knowledge base."""

    knowledge = {
        "RAG": (
            "Retrieval-Augmented Generation retrieves relevant external "
            "information and supplies it to an LLM as context."
        ),
        "Embeddings": (
            "Embeddings represent text as numerical vectors. "
            "They enable semantic similarity search."
        ),
        "Agents": (
            "AI agents use an LLM to select actions, call tools, "
            "observe results, and decide what to do next."
        ),
        "MCP": (
            "Model Context Protocol provides a standardized way for "
            "AI applications to connect with external tools and data."
        ),
    }

    query_lower = query.lower()

    matches = [
        {"topic": topic, "content": content}
        for topic, content in knowledge.items()
        if topic.lower() in query_lower
        or any(
            word in content.lower()
            for word in query_lower.split()
            if len(word) > 3
        )
    ]

    return {
        "results": matches[:3],
        "count": len(matches[:3]),
    }


# ---------------------------------------------------------
# 3. Register tools
# ---------------------------------------------------------

TOOLS = {
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. 25 * 4",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search technical knowledge about RAG, embeddings, agents, or MCP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Technical search query",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

# ---------------------------------------------------------
# 4. Response extraction
# ---------------------------------------------------------

def extract_text(value):
    """Extract user-facing text without exposing reasoning blocks."""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = [extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)

    if isinstance(value, dict):
        if value.get("type") in ("reasoning", "thinking"):
            return ""

        for key in ("text", "output_text"):
            if isinstance(value.get(key), str):
                return value[key]

        for key in ("content", "parts"):
            if key in value:
                return extract_text(value[key])

        return ""

    block_type = getattr(value, "type", None)

    if block_type in ("reasoning", "thinking"):
        return ""

    for attribute in ("text", "output_text", "content"):
        item = getattr(value, attribute, None)
        if item is not None:
            result = extract_text(item)
            if result:
                return result

    return ""


# ---------------------------------------------------------
# 5. Agent execution loop
# ---------------------------------------------------------

def run_agent(user_query: str, max_iterations: int = 5):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical assistant with access to tools. "
                "Use tools when necessary. "
                "Do not invent tool results. "
                "After receiving tool results, either call another tool "
                "or provide a concise final answer."
            ),
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    for iteration in range(max_iterations):

        response = client.chat.completions.create(
            model=MODEL_ENDPOINT,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=700,
        )

        message = response.choices[0].message

        # Inspect tool calls before attempting text extraction.
        tool_calls = message.tool_calls or []

        if not tool_calls:
            answer = extract_text(message.content)

            if answer:
                return answer

            print("No recognizable user-facing text in model response.")
            print("Content type:", type(message.content).__name__)
            print("Finish reason:", response.choices[0].finish_reason)
            return None

        # Preserve the assistant tool-call message in conversation history.
        messages.append(message.model_dump(exclude_none=True))

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)

                if tool_name not in TOOLS:
                    raise ValueError(f"Unknown tool: {tool_name}")

                result = TOOLS[tool_name](**arguments)

            except Exception as exc:
                result = {"error": str(exc)}

            print(f"\nIteration {iteration + 1}")
            print("Tool:", tool_name)
            print("Result:", result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    return "Agent stopped: maximum iteration limit reached."


# ---------------------------------------------------------
# 6. Run experiments
# ---------------------------------------------------------

print("\nEXPERIMENT 1: Tool selection")
print(run_agent("What is 125 * 48?"))

print("\nEXPERIMENT 2: Knowledge retrieval")
print(run_agent("Explain how RAG works."))

print("\nEXPERIMENT 3: Multi-step reasoning")
print(
    run_agent(
        "Find out what embeddings are, then calculate "
        "the result of multiplying 128 by 64."
    )
)

# ---------------------------------------------------------
# 7. Engineering observations
# ---------------------------------------------------------

print("\nENGINEERING OBSERVATIONS")
print("- The LLM selects tools; application code executes them.")
print("- Tool outputs return to the model as observations.")
print("- The application controls iteration limits.")
print("- Tool arguments and permissions require validation.")
print("- Tool calling does not make an agent autonomous or safe by itself.")
