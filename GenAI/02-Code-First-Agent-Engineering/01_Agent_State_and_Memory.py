
"""
Agent State and Memory
Experiments:
1. Conversation state management
2. Working memory extraction
3. Persistent memory with Delta tables
4. Memory retrieval and injection into prompts
5. State isolation between conversations
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# 1. Configuration
# ============================================================

from databricks.sdk import WorkspaceClient
from openai import OpenAI

MODEL_ENDPOINT = "system.ai.qwen35-122b-a10b"

workspace = WorkspaceClient()

client = OpenAI(
    api_key=workspace.config.authenticate()["Authorization"].replace(
        "Bearer ", ""
    ),
    base_url=(
        f"https://{workspace.config.host}"
        "/ai-gateway/mlflow/v1"
    ),
)

current_user = spark.sql(
    "SELECT current_user()"
).first()[0]

MEMORY_TABLE = "agent_memory"


# ============================================================
# 2. LLM Response Extraction
# ============================================================

def extract_text(value: Any) -> str:
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

    if getattr(value, "type", None) in ("reasoning", "thinking"):
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
        max_tokens=700,
    )

    message = response.choices[0].message
    return extract_text(message.content)


# ============================================================
# 3. Structured Memory Model
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
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages = []
        self.working_memory = {}

    def add_message(self, role, content):
        if role not in ("user", "assistant"):
            raise ValueError("Unsupported conversation role.")

        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def add_memory(self, memory: MemoryItem):
        self.working_memory[memory.key] = memory.model_dump()

    def get_history(self):
        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in self.messages
        ]

    def summary(self):
        return {
            "conversation_id": self.conversation_id,
            "message_count": len(self.messages),
            "working_memory": self.working_memory,
        }


# ============================================================
# 5. Persistent Memory Storage
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


def save_memory(user_id, memory: MemoryItem):
    """
    Save a memory or update the existing active memory
    with the same user_id and memory_key.
    """

    from delta.tables import DeltaTable

    memory_id = str(uuid.uuid4())

    source = spark.createDataFrame(
        [(
            memory_id,
            user_id,
            memory.key,
            memory.value,
            memory.category,
            memory.confidence,
        )],
        schema=(
            "memory_id STRING, "
            "user_id STRING, "
            "memory_key STRING, "
            "memory_value STRING, "
            "category STRING, "
            "confidence DOUBLE"
        ),
    )

    now = datetime.now(timezone.utc)

    if not spark.catalog.tableExists(MEMORY_TABLE):
        create_memory_table()

    target = DeltaTable.forName(spark, MEMORY_TABLE)

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
    rows = (
        spark.table(MEMORY_TABLE)
        .filter(
            (spark.table(MEMORY_TABLE).user_id == user_id)
            & (spark.table(MEMORY_TABLE).is_active == True)
        )
        .orderBy("updated_at", ascending=False)
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
    """
    Ask the LLM to identify useful, explicit, reusable facts.
    Treat extracted memories as candidates, not trusted facts.
    """

    prompt = """
You are a memory extraction component.

Extract only explicit, useful, reusable facts from the
conversation.

Rules:
- Do not invent facts.
- Do not infer personal characteristics.
- Ignore temporary requests and casual conversation.
- Do not store passwords, secrets, credentials, or tokens.
- Prefer stable preferences, project facts, and decisions.
- Return an empty memories list if nothing is worth retaining.

Return JSON in this format:
{
  "memories": [
    {
      "key": "preferred_language",
      "value": "English",
      "category": "preference",
      "confidence": 0.99
    }
  ]
}
"""

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(conversation_history),
        },
    ]

    raw = ask_llm(messages)

    try:
        data = json.loads(raw)
        extracted = MemoryExtraction.model_validate(data)
        return extracted.memories
    except Exception as error:
        print("Memory extraction failed:", str(error))
        print("Model output:", raw[:1000])
        return []


# ============================================================
# 7. Memory Injection
# ============================================================

def build_context(memories):
    if not memories:
        return "No relevant stored memories."

    return "\n".join(
        f"- {item['key']}: {item['value']}"
        for item in memories
    )


def agent_turn(state, user_message, user_id):
    """
    Execute one agent turn using conversation history
    and previously stored memories.
    """

    state.add_message("user", user_message)

    persistent_memories = retrieve_memories(user_id)

    system_prompt = f"""
You are a helpful AI assistant.

Use the following stored user context when relevant:

{build_context(persistent_memories)}

Rules:
- Do not claim that stored information is always current.
- Prefer the latest information provided in the conversation.
- Do not invent personal details.
- If memory conflicts with the user's latest statement,
  prioritize the latest statement.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        *state.get_history(),
    ]

    answer = ask_llm(messages)

    if not answer:
        answer = (
            "I could not generate a response. "
            "Please try again."
        )

    state.add_message("assistant", answer)

    return answer


# ============================================================
# 8. Experiment: Conversation and Working Memory
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
# 9. Experiment: Extract and Persist Memory
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 2: MEMORY EXTRACTION")
print("=" * 60)

create_memory_table()

candidate_memories = extract_memories(state.get_history())

for memory in candidate_memories:
    print(memory.model_dump())

    # Demonstration only: in production, add a policy
    # and user-consent gate before persisting candidates.
    if memory.confidence >= 0.80:
        save_memory(current_user, memory)

print("\nStored memories:")
stored_memories = retrieve_memories(current_user)

print(json.dumps(stored_memories, indent=2))


# ============================================================
# 10. Experiment: Memory Across Conversations
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 3: CROSS-CONVERSATION MEMORY")
print("=" * 60)

# New state simulates a separate conversation/session.
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
# 11. Experiment: State Isolation
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

print("\nState isolation verified.")


# ============================================================
# 12. Final Summary
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)

print("Conversation state: implemented")
print("Working memory: implemented")
print("Persistent Delta memory: implemented")
print("Memory retrieval: implemented")
print("Cross-conversation context: demonstrated")
print("Conversation isolation: verified")

print("\nMemory table:", MEMORY_TABLE)
print("User scope:", current_user)
