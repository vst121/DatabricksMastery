
"""
MCP Server and Client

Experiments:
1. MCP server implementation
2. JSON-RPC 2.0 messaging
3. MCP initialization
4. Tool discovery
5. Tool invocation
6. MCP-to-LLM tool adapter
7. Agent execution through MCP
"""

import json
import uuid
import time
import ast
import math
import operator

from typing import Any
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
# 3. Typed Tool Definitions
# ============================================================

class CalculatorInput(BaseModel):

    expression: str = Field(
        description="Mathematical expression.",
        min_length=1,
        max_length=200,
    )


class SalesInput(BaseModel):

    country: str = Field(
        description="Country to analyze.",
    )

    metric: str = Field(
        default="revenue",
        description="Metric: revenue, profit, or units_sold.",
    )


# ============================================================
# 4. Tool Implementations
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


def calculate(expression: str):

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
                raise ValueError("Unsupported operator.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent too large.")

            result = operation(left, right)

            if not math.isfinite(float(result)):
                raise ValueError("Invalid numerical result.")

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


def get_sales_summary(
    country: str,
    metric: str,
):

    if country not in ("Germany", "France"):
        raise ValueError("Unsupported country.")

    if metric not in ("revenue", "profit", "units_sold"):
        raise ValueError("Unsupported metric.")

    if not spark.catalog.tableExists("genai_sales"):
        raise ValueError(
            "genai_sales table is unavailable. "
            "Run the earlier sales-data experiment first."
        )

    from pyspark.sql import functions as F

    aggregations = {
        "revenue": F.sum("revenue"),
        "profit": F.sum("profit"),
        "units_sold": F.sum("quantity"),
    }

    result = (
        spark.table("genai_sales")
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
# 5. MCP Server
# ============================================================

MCP_PROTOCOL_VERSION = "2025-06-18"


class MCPServer:

    def __init__(self, name: str, version: str):

        self.name = name
        self.version = version
        self.initialized = False
        self.tools = {}
        self.execution_log = []

    def register_tool(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        function,
    ):

        if name in self.tools:
            raise ValueError(f"Duplicate tool: {name}")

        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_model.model_json_schema(),
            "input_model": input_model,
            "function": function,
        }

    def list_tools(self):

        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            }
            for tool in self.tools.values()
        ]

    def call_tool(self, name: str, arguments: dict):

        if not self.initialized:
            raise RuntimeError("MCP session is not initialized.")

        if name not in self.tools:
            raise ValueError(f"Unknown MCP tool: {name}")

        tool = self.tools[name]

        validated = tool["input_model"].model_validate(
            arguments
        )

        started = time.perf_counter()

        result = tool["function"](**validated.model_dump())

        elapsed = (time.perf_counter() - started) * 1000

        self.execution_log.append({
            "tool": name,
            "arguments": arguments,
            "execution_time_ms": elapsed,
        })

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, default=str),
                }
            ],
            "isError": False,
        }

    def handle(self, request: dict):

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        def success(result):
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }

        def failure(code, message):
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": code,
                    "message": message,
                },
            }

        try:

            if request.get("jsonrpc") != "2.0":
                return failure(-32600, "Invalid JSON-RPC version.")

            if method == "initialize":

                requested_version = params.get("protocolVersion")

                if not requested_version:
                    return failure(-32602, "Missing protocol version.")

                self.initialized = True

                return success({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {
                            "listChanged": False
                        }
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version,
                    },
                })

            if method == "notifications/initialized":
                return None

            if method == "ping":
                return success({})

            if not self.initialized:
                return failure(
                    -32002,
                    "Server has not been initialized."
                )

            if method == "tools/list":
                return success({
                    "tools": self.list_tools()
                })

            if method == "tools/call":

                name = params.get("name")
                arguments = params.get("arguments", {})

                result = self.call_tool(name, arguments)

                return success(result)

            return failure(
                -32601,
                f"Method not found: {method}"
            )

        except ValidationError as error:

            return failure(
                -32602,
                str(error)
            )

        except Exception as error:

            return failure(
                -32000,
                str(error)
            )


# ============================================================
# 6. Register MCP Tools
# ============================================================

server = MCPServer(
    name="Databricks Analytics MCP Server",
    version="1.0.0",
)

server.register_tool(
    name="calculator",
    description="Evaluate a safe mathematical expression.",
    input_model=CalculatorInput,
    function=lambda expression: {
        "expression": expression,
        "result": calculate(expression),
    },
)

server.register_tool(
    name="sales_summary",
    description="Retrieve sales metrics for a country.",
    input_model=SalesInput,
    function=get_sales_summary,
)

print("Registered MCP tools:", len(server.tools))


# ============================================================
# 7. In-Memory Transport
# ============================================================

class InMemoryTransport:

    def __init__(self, server: MCPServer):
        self.server = server

    def send(self, message: dict):

        # Simulates request/response transport.
        # No subprocess or external service is created.

        return self.server.handle(message)


transport = InMemoryTransport(server)


# ============================================================
# 8. MCP Client
# ============================================================

class MCPClient:

    def __init__(self, transport: InMemoryTransport):

        self.transport = transport
        self.initialized = False
        self.server_info = None

    def request(self, method: str, params=None):

        message = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        }

        response = self.transport.send(message)

        if response is None:
            return None

        if "error" in response:
            raise RuntimeError(
                response["error"]["message"]
            )

        return response.get("result")

    def initialize(self):

        result = self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "Databricks MCP Client",
                    "version": "1.0.0",
                },
            },
        )

        self.server_info = result["serverInfo"]

        self.transport.send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

        self.initialized = True

        return result

    def list_tools(self):

        if not self.initialized:
            raise RuntimeError("Client is not initialized.")

        result = self.request("tools/list")

        return result["tools"]

    def call_tool(self, name: str, arguments: dict):

        if not self.initialized:
            raise RuntimeError("Client is not initialized.")

        return self.request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )


mcp_client = MCPClient(transport)


# ============================================================
# 9. Experiment 1: MCP Initialization
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 1: MCP INITIALIZATION")
print("=" * 60)

initialization = mcp_client.initialize()

print(
    json.dumps(
        initialization,
        indent=2,
    )
)


# ============================================================
# 10. Experiment 2: Tool Discovery
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 2: TOOL DISCOVERY")
print("=" * 60)

discovered_tools = mcp_client.list_tools()

for tool in discovered_tools:

    print({
        "name": tool["name"],
        "description": tool["description"],
    })


# ============================================================
# 11. Experiment 3: Direct MCP Tool Calls
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 3: DIRECT MCP TOOL CALLS")
print("=" * 60)

calculator_result = mcp_client.call_tool(
    "calculator",
    {
        "expression": "(25 * 4) + 10"
    },
)

print("Calculator:", calculator_result)

sales_result = mcp_client.call_tool(
    "sales_summary",
    {
        "country": "Germany",
        "metric": "revenue",
    },
)

print("Sales:", sales_result)


# ============================================================
# 12. Experiment 4: MCP to OpenAI Tool Adapter
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 4: MCP TOOL ADAPTER")
print("=" * 60)


def mcp_tools_to_openai(tools):

    converted = []

    for tool in tools:

        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        })

    return converted


llm_tools = mcp_tools_to_openai(
    discovered_tools
)

print(
    json.dumps(
        llm_tools,
        indent=2,
    )
)


# ============================================================
# 13. Experiment 5: Qwen Agent Using MCP
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 5: QWEN AGENT WITH MCP")
print("=" * 60)

messages = [
    {
        "role": "system",
        "content": (
            "You are a data analysis assistant. "
            "Use the available tools when needed. "
            "Never invent tool results."
        ),
    },
    {
        "role": "user",
        "content": (
            "Calculate 125 multiplied by 8. "
            "Then retrieve Germany's revenue."
        ),
    },
]

for iteration in range(5):

    response = client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=messages,
        tools=llm_tools,
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

        tool_name = tool_call.function.name

        try:
            arguments = json.loads(
                tool_call.function.arguments
            )

            result = mcp_client.call_tool(
                tool_name,
                arguments,
            )

        except Exception as error:

            result = {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "error": str(error)
                        }),
                    }
                ],
                "isError": True,
            }

        print("\nMCP tool:", tool_name)
        print("MCP result:", result)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, default=str),
        })

else:
    print("Maximum agent iterations reached.")


# ============================================================
# 14. Experiment 6: MCP Execution Log
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 6: MCP EXECUTION LOG")
print("=" * 60)

print(
    json.dumps(
        server.execution_log,
        indent=2,
        default=str,
    )
)


# ============================================================
# 15. Final Summary
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)

print("MCP server: implemented")
print("JSON-RPC messaging: implemented")
print("Initialization: implemented")
print("Tool discovery: implemented")
print("Tool invocation: implemented")
print("MCP client: implemented")
print("LLM tool adapter: implemented")
print("Qwen agent integration: demonstrated")
print("Execution logging: implemented")
