
"""
Typed Tool Engineering

Experiments:
1. Pydantic tool input schemas
2. Automatic JSON Schema generation
3. Centralized tool registry
4. Input validation
5. Authorization policies
6. Controlled tool execution
7. LLM-driven tool selection
8. Tool execution tracing
"""

import ast
import json
import math
import operator
import time
import uuid

from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from databricks.sdk import WorkspaceClient


# ============================================================
# 1. Configuration
# ============================================================

MODEL_ENDPOINT = "system.ai.qwen35-122b-a10b"

workspace = WorkspaceClient()
headers = workspace.config.authenticate()

token = headers["Authorization"].replace("Bearer ", "")
workspace_url = workspace.config.host.rstrip("/")

client = OpenAI(
    api_key=token,
    base_url=f"{workspace_url}/ai-gateway/mlflow/v1",
    timeout=60.0,
    max_retries=2,
)

print("Model endpoint:", MODEL_ENDPOINT)


# ============================================================
# 2. LLM Utilities
# ============================================================

def extract_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts = [extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)

    if isinstance(value, dict):

        if value.get("type") in ("reasoning", "thinking"):
            return ""

        for key in ("text", "output_text"):
            if isinstance(value.get(key), str):
                return value[key].strip()

        for key in ("content", "parts"):
            if key in value:
                return extract_text(value[key])

    if getattr(value, "type", None) in ("reasoning", "thinking"):
        return ""

    for attribute in ("text", "output_text", "content"):
        item = getattr(value, attribute, None)

        if item is not None:
            result = extract_text(item)
            if result:
                return result

    return ""


# ============================================================
# 3. Typed Tool Input Schemas
# ============================================================

class CalculatorInput(BaseModel):

    expression: str = Field(
        description="A mathematical expression.",
        min_length=1,
        max_length=200,
    )


class SalesSummaryInput(BaseModel):

    country: str = Field(
        description="Country to analyze.",
    )

    metric: str = Field(
        default="revenue",
        description="Metric: revenue, profit, or units_sold.",
    )


class ToolResult(BaseModel):

    success: bool
    tool_name: str
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


# ============================================================
# 4. Safe Calculator
# ============================================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate_expression(expression: str):

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):

        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError("Only numeric constants are allowed.")

            return node.value

        if isinstance(node, ast.BinOp):

            operation = OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported mathematical operator.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large.")

            result = operation(left, right)

            if not math.isfinite(float(result)):
                raise ValueError("Result is not finite.")

            if abs(result) > 1e15:
                raise ValueError("Result exceeds allowed magnitude.")

            return result

        if isinstance(node, ast.UnaryOp):

            operation = OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported unary operator.")

            return operation(evaluate(node.operand))

        raise ValueError("Unsupported expression.")

    return evaluate(tree)


# ============================================================
# 5. Sales Analytics Tool
# ============================================================

def sales_summary(
    country: str,
    metric: str,
):

    allowed_countries = ["Germany", "France"]
    allowed_metrics = ["revenue", "profit", "units_sold"]

    if country not in allowed_countries:
        raise ValueError("Unsupported country.")

    if metric not in allowed_metrics:
        raise ValueError("Unsupported metric.")

    table = "genai_sales"

    if not spark.catalog.tableExists(table):
        raise ValueError(
            "The genai_sales table does not exist. "
            "Run the earlier sales-data experiment first."
        )

    from pyspark.sql import functions as F

    aggregations = {
        "revenue": F.sum("revenue"),
        "profit": F.sum("profit"),
        "units_sold": F.sum("quantity"),
    }

    result = (
        spark.table(table)
        .filter(F.col("country") == country)
        .agg(
            aggregations[metric].alias(metric)
        )
        .first()
    )

    return {
        "country": country,
        "metric": metric,
        "value": float(result[metric] or 0),
    }


# ============================================================
# 6. Tool Definition
# ============================================================

class ToolDefinition:

    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        function: Callable,
        allowed_roles: set[str],
    ):

        self.name = name
        self.description = description
        self.input_model = input_model
        self.function = function
        self.allowed_roles = allowed_roles

    def openai_schema(self):

        schema = self.input_model.model_json_schema()

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# ============================================================
# 7. Tool Registry
# ============================================================

class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register(self, tool: ToolDefinition):

        if tool.name in self.tools:
            raise ValueError(
                f"Tool already registered: {tool.name}"
            )

        self.tools[tool.name] = tool

    def get(self, name: str):

        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")

        return self.tools[name]

    def schemas(self):

        return [
            tool.openai_schema()
            for tool in self.tools.values()
        ]


registry = ToolRegistry()

registry.register(
    ToolDefinition(
        name="calculator",
        description="Evaluate a safe mathematical expression.",
        input_model=CalculatorInput,
        function=lambda data: evaluate_expression(
            data.expression
        ),
        allowed_roles={"user", "analyst"},
    )
)

registry.register(
    ToolDefinition(
        name="sales_summary",
        description="Retrieve an aggregated sales metric for a country.",
        input_model=SalesSummaryInput,
        function=lambda data: sales_summary(
            data.country,
            data.metric,
        ),
        allowed_roles={"analyst"},
    )
)


# ============================================================
# 8. Execution Gateway
# ============================================================

class ToolExecutor:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.traces = []

    def execute(
        self,
        name: str,
        arguments: dict,
        role: str = "analyst",
    ) -> ToolResult:

        started = time.perf_counter()

        trace_id = str(uuid.uuid4())

        try:

            tool = self.registry.get(name)

            if role not in tool.allowed_roles:
                raise PermissionError(
                    f"Role '{role}' cannot execute '{name}'."
                )

            validated_input = tool.input_model.model_validate(
                arguments
            )

            result = tool.function(validated_input)

            execution = ToolResult(
                success=True,
                tool_name=name,
                result=result,
                execution_time_ms=(
                    time.perf_counter() - started
                ) * 1000,
            )

        except (
            ValidationError,
            ValueError,
            PermissionError,
            ArithmeticError,
            Exception,
        ) as error:

            execution = ToolResult(
                success=False,
                tool_name=name,
                error=str(error),
                execution_time_ms=(
                    time.perf_counter() - started
                ) * 1000,
            )

        self.traces.append({
            "trace_id": trace_id,
            **execution.model_dump(),
        })

        return execution


executor = ToolExecutor(registry)


# ============================================================
# 9. Experiment: Direct Tool Execution
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 1: DIRECT TOOL EXECUTION")
print("=" * 60)

calculator_result = executor.execute(
    "calculator",
    {"expression": "(25 * 4) + 10"},
    role="analyst",
)

print(calculator_result.model_dump())

sales_result = executor.execute(
    "sales_summary",
    {
        "country": "Germany",
        "metric": "revenue",
    },
    role="analyst",
)

print(sales_result.model_dump())


# ============================================================
# 10. Experiment: Validation and Authorization
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 2: VALIDATION AND AUTHORIZATION")
print("=" * 60)

invalid_input = executor.execute(
    "calculator",
    {"expression": "import os"},
    role="analyst",
)

print("Invalid input:", invalid_input.model_dump())

unauthorized = executor.execute(
    "sales_summary",
    {
        "country": "Germany",
        "metric": "profit",
    },
    role="user",
)

print("Unauthorized:", unauthorized.model_dump())


# ============================================================
# 11. Experiment: Tool Schema Generation
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 3: GENERATED TOOL SCHEMAS")
print("=" * 60)

print(
    json.dumps(
        registry.schemas(),
        indent=2,
    )
)


# ============================================================
# 12. Experiment: LLM Tool Calling
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 4: LLM TOOL CALLING")
print("=" * 60)

messages = [
    {
        "role": "system",
        "content": (
            "You are an analytical assistant. "
            "Use tools when required. "
            "Never invent tool results."
        ),
    },
    {
        "role": "user",
        "content": (
            "Calculate 125 multiplied by 8, "
            "then tell me the revenue in Germany."
        ),
    },
]

for iteration in range(5):

    response = client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=messages,
        tools=registry.schemas(),
        tool_choice="auto",
        temperature=0.1,
        max_tokens=1200,
    )

    choice = response.choices[0]
    message = choice.message

    if not message.tool_calls:

        answer = extract_text(message.content)

        print(
            "\nAssistant:",
            answer or "No extractable text returned."
        )

        if not answer:
            print("Finish reason:", choice.finish_reason)
            print("Content:", repr(message.content)[:2000])

        break

    messages.append(
        message.model_dump(exclude_none=True)
    )

    for tool_call in message.tool_calls:

        try:
            arguments = json.loads(
                tool_call.function.arguments
            )

        except json.JSONDecodeError:

            result = ToolResult(
                success=False,
                tool_name=tool_call.function.name,
                error="Invalid JSON tool arguments.",
            )

        else:

            result = executor.execute(
                tool_call.function.name,
                arguments,
                role="analyst",
            )

        print(
            "\nTool:",
            tool_call.function.name
        )

        print(
            "Result:",
            result.model_dump()
        )

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(
                result.model_dump(),
                default=str,
            ),
        })

else:
    print("Maximum agent iterations reached.")


# ============================================================
# 13. Experiment: Execution Traces
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 5: TOOL EXECUTION TRACES")
print("=" * 60)

print(
    json.dumps(
        executor.traces,
        indent=2,
        default=str,
    )
)


# ============================================================
# 14. Final Summary
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)

print("Typed input schemas: implemented")
print("JSON Schema generation: implemented")
print("Tool registry: implemented")
print("Input validation: implemented")
print("Authorization: implemented")
print("Centralized execution: implemented")
print("LLM tool calling: demonstrated")
print("Execution tracing: implemented")
