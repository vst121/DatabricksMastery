
"""
Agent State and Memory

Experiments:
1. Conversation state management
2. Working memory
3. Persistent memory with Delta Lake
4. Memory extraction using an LLM
5. Memory retrieval and prompt injection
6. Cross-conversation memory
7. Conversation isolation
"""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field
from pyspark.sql import functions as F
from delta.tables import DeltaTable
from databricks.sdk import WorkspaceClient


# ============================================================
# 1. Configuration and Authentication
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

current_user = spark.sql(
    "SELECT current_user()"
).first()[0]

context = spark.sql("""
    SELECT current_catalog() AS catalog,
           current_schema() AS schema
""").first()

CATALOG = context["catalog"]
SCHEMA = context["schema"]

def quote_identifier(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid SQL identifier: {value}")
    return f"`{value}`"

MEMORY_TABLE = (
    f"{quote_identifier(CATALOG)}."
    f"{quote_identifier(SCHEMA)}."
    f"`agent_memory`"
)

print("Workspace:", workspace_url)
print("Current user:", current_user)
print("Memory table:", MEMORY_TABLE)


# ============================================================
# 2. LLM Utilities
# ============================================================


def extract_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts = [
            extract_text(item)
            for item in value
        ]
        return "\n".join(part for part in parts if part)

    if isinstance(value, dict):

        block_type = value.get("type")

        if block_type in ("reasoning", "thinking"):
            return ""

        for key in ("text", "output_text"):
            item = value.get(key)

            if isinstance(item, str) and item.strip():
                return item.strip()

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


def ask_llm(messages, temperature=0.2):

    response = client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=messages,
        temperature=temperature,
        max_tokens=1200,
    )

    choice = response.choices[0]
    message = choice.message

    answer = extract_text(message.content)

    if answer:
        return answer

    # Diagnostic information without exposing model reasoning.
    print("\nLLM RESPONSE DIAGNOSTICS")
    print("Response ID:", response.id)
    print("Model:", response.model)
    print("Finish reason:", choice.finish_reason)
    print("Message role:", message.role)
    print("Content type:", type(message.content).__name__)
    print("Content representation:", repr(message.content)[:2000])
    print("Refusal:", getattr(message, "refusal", None))
    print(
        "Tool calls:",
        len(message.tool_calls or [])
        if message.tool_calls else 0
    )

    return ""


def parse_json_response(raw):

    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

    return json.loads(cleaned)


# ============================================================
# 3. Memory Data Models
# ============================================================

class MemoryItem(BaseModel):

    key: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=1000)
    category: str = "general"
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryExtraction(BaseModel):

    memories: list[MemoryItem] = Field(default_factory=list)


# ============================================================
# 4. Conversation State
# ============================================================

class AgentState:

    def __init__(self, conversation_id=None):

        self.conversation_id = (
            conversation_id or str(uuid.uuid4())
        )

        self.messages = []
        self.working_memory = {}

    def add_message(self, role, content):

        if role not in ("user", "assistant"):
            raise ValueError("Unsupported conversation role.")

        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        })

    def add_memory(self, memory: MemoryItem):

        self.working_memory[memory.key] = (
            memory.model_dump()
        )

    def get_history(self):

        return [
            {
                "role": item["role"],
                "content": item["content"],
            }
            for item in self.messages
        ]

    def summary(self):

        return {
            "conversation_id": self.conversation_id,
            "message_count": len(self.messages),
            "working_memory": self.working_memory,
        }


# ============================================================
# 5. Persistent Memory: Delta Lake
# ============================================================

def create_memory_table():

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {MEMORY_TABLE} (
            memory_id STRING,
            user_id STRING,
            memory_key STRING,
            memory_value STRING,
            category STRING,
            confidence DOUBLE,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            is_active BOOLEAN
        )
        USING DELTA
    """)

    print("Memory table ready.")


def save_memory(user_id, memory: MemoryItem):

    if not spark.catalog.tableExists(
        f"{CATALOG}.{SCHEMA}.agent_memory"
    ):
        create_memory_table()

    source = spark.createDataFrame(
        [(
            str(uuid.uuid4()),
            user_id,
            memory.key,
            memory.value,
            memory.category,
            memory.confidence,
        )],
        schema="""
            memory_id STRING,
            user_id STRING,
            memory_key STRING,
            memory_value STRING,
            category STRING,
            confidence DOUBLE
        """,
    )

    target = DeltaTable.forName(
        spark,
        f"{CATALOG}.{SCHEMA}.agent_memory",
    )

    (
        target.alias("target")
        .merge(
            source.alias("source"),
            """
            target.user_id = source.user_id
            AND target.memory_key = source.memory_key
            AND target.is_active = true
            """
        )
        .whenMatchedUpdate(set={
            "memory_value": "source.memory_value",
            "category": "source.category",
            "confidence": "source.confidence",
            "updated_at": "current_timestamp()",
        })
        .whenNotMatchedInsert(values={
            "memory_id": "source.memory_id",
            "user_id": "source.user_id",
            "memory_key": "source.memory_key",
            "memory_value": "source.memory_value",
            "category": "source.category",
            "confidence": "source.confidence",
            "created_at": "current_timestamp()",
            "updated_at": "current_timestamp()",
            "is_active": "true",
        })
        .execute()
    )


def retrieve_memories(user_id, limit=20):

    if not spark.catalog.tableExists(
        f"{CATALOG}.{SCHEMA}.agent_memory"
    ):
        return []

    rows = (
        spark.table(f"{CATALOG}.{SCHEMA}.agent_memory")
        .filter(
            (F.col("user_id") == user_id)
            & (F.col("is_active") == True)
        )
        .orderBy(
            F.col("updated_at").desc()
        )
        .limit(limit)
        .collect()
    )

    return [
        {
            "key": row.memory_key,
            "value": row.memory_value,
            "category": row.category,
            "confidence": row.confidence,
        }
        for row in rows
    ]


# ============================================================
# 6. Memory Extraction
# ============================================================

def extract_memories(conversation_history):

    system_prompt = """
You are a memory extraction component.

Extract explicit, useful and reusable facts.

Rules:
- Never invent facts.
- Do not infer personal characteristics.
- Ignore temporary requests and casual conversation.
- Never store passwords, credentials, API keys or secrets.
- Prefer stable project facts, preferences and decisions.
- Return an empty memories list if nothing is worth retaining.

Return valid JSON only:

{
  "memories": [
    {
      "key": "project_backend",
      "value": "Python with FastAPI",
      "category": "project",
      "confidence": 0.99
    }
  ]
}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": json.dumps(conversation_history),
        },
    ]

    try:

        raw = ask_llm(messages)
        data = parse_json_response(raw)

        result = MemoryExtraction.model_validate(data)

        return result.memories

    except Exception as error:

        print("Memory extraction failed:", str(error))
        return []


# ============================================================
# 7. Memory Context Construction
# ============================================================

def build_memory_context(memories):

    if not memories:
        return "No relevant stored memories."

    return "\n".join(
        f"- {item['key']}: {item['value']}"
        for item in memories
    )


# ============================================================
# 8. Agent Execution
# ============================================================

def agent_turn(state, user_message, user_id):

    state.add_message("user", user_message)

    memories = retrieve_memories(user_id)

    system_prompt = f"""
You are a helpful AI assistant.

Relevant stored context:

{build_memory_context(memories)}

Instructions:
- Use stored context when relevant.
- Never assume stored information is always current.
- Prioritize the latest user statement.
- Do not invent personal details.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *state.get_history(),
    ]

    answer = ask_llm(messages)

    state.add_message("assistant", answer)

    return answer


# ============================================================
# 9. Experiment 1: Conversation State
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 1: CONVERSATION STATE")
print("=" * 60)

state = AgentState()

state.add_message(
    "user",
    "I am building a customer support AI assistant."
)

state.add_message(
    "assistant",
    "We can design it with retrieval and tools."
)

state.add_message(
    "user",
    "The backend is written in Python with FastAPI."
)

print(json.dumps(state.summary(), indent=2))


# ============================================================
# 10. Experiment 2: Extract and Save Memory
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 2: MEMORY EXTRACTION")
print("=" * 60)

create_memory_table()

candidate_memories = extract_memories(
    state.get_history()
)

print("Candidate memories:", len(candidate_memories))

for memory in candidate_memories:

    print(memory.model_dump())

    # Demonstration policy.
    # Production systems should require consent and
    # validation before persisting extracted memories.

    if memory.confidence >= 0.80:
        save_memory(current_user, memory)

print("\nPersistent memories:")

stored_memories = retrieve_memories(current_user)

print(json.dumps(stored_memories, indent=2))


# ============================================================
# 11. Experiment 3: Cross-Conversation Memory
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 3: CROSS-CONVERSATION MEMORY")
print("=" * 60)

# Simulate a completely new conversation.

new_state = AgentState()

question = (
    "What do you know about the customer support "
    "assistant project I am building?"
)

answer = agent_turn(
    state=new_state,
    user_message=question,
    user_id=current_user,
)

print("\nUser:", question)
print("\nAssistant:", answer)


# ============================================================
# 12. Experiment 4: Conversation Isolation
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 4: CONVERSATION ISOLATION")
print("=" * 60)

another_state = AgentState()

another_state.add_message(
    "user",
    "I am starting a different project."
)

print("Conversation A:", state.conversation_id)
print("Conversation B:", another_state.conversation_id)

print("Conversation A messages:", len(state.messages))
print("Conversation B messages:", len(another_state.messages))

assert state.conversation_id != another_state.conversation_id
assert len(another_state.messages) == 1

print("\nConversation isolation verified.")


# ============================================================
# 13. Final Summary
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)

print("Conversation state: implemented")
print("Working memory: implemented")
print("Persistent Delta memory: implemented")
print("Memory extraction: implemented")
print("Memory retrieval: implemented")
print("Cross-conversation context: demonstrated")
print("Conversation isolation: verified")

print("\nMemory table:", MEMORY_TABLE)
print("User scope:", current_user)
