"""Repository for agent_messages."""

from __future__ import annotations

from backend.persistence.database import Database
from backend.schemas.messages import AgentMessage


class MessageRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, msg: AgentMessage) -> None:
        import json
        self._db.connection.execute(
            """INSERT INTO agent_messages
               (message_id, agent_id, content, structured_outputs, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (
                msg.message_id,
                msg.agent_id,
                msg.content,
                json.dumps(msg.structured_outputs),
                msg.timestamp.isoformat(),
            ),
        )
        self._db.connection.commit()

    def list_by_agent(self, agent_id: str) -> list[AgentMessage]:
        import json
        rows = self._db.connection.execute(
            "SELECT * FROM agent_messages WHERE agent_id = ?", (agent_id,)
        ).fetchall()
        return [
            AgentMessage(
                message_id=r["message_id"],
                agent_id=r["agent_id"],
                content=r["content"],
                structured_outputs=json.loads(r["structured_outputs"]),
                timestamp=r["timestamp"],
            )
            for r in rows
        ]
