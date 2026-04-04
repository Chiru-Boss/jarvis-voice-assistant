import json
import os
from datetime import datetime


class ConversationMemory:
    """Persistent conversation memory backed by a JSON file."""

    def __init__(self, memory_file='jarvis_memory.json', max_history=10):
        self.memory_file = memory_file
        self.max_history = max_history
        self.memory = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        """Load conversation history from disk (if available)."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.memory = data.get('conversations', [])
            except (json.JSONDecodeError, IOError):
                self.memory = []

    def save(self):
        """Persist conversation history to disk."""
        try:
            data = {'conversations': self.memory}
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️  Could not save memory: {e}")

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def add_conversation(self, user_input, bot_response):
        """Add a user/assistant exchange and auto-save."""
        self.memory.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
        })
        self.save()

    def get_recent_messages(self, n=None):
        """Return recent exchanges as OpenAI-style message dicts."""
        limit = n if n is not None else self.max_history
        recent = self.memory[-limit:]
        messages = []
        for entry in recent:
            messages.append({'role': 'user', 'content': entry['user_input']})
            messages.append({'role': 'assistant', 'content': entry['bot_response']})
        return messages

    def get_memory(self):
        return self.memory

    def clear_memory(self):
        self.memory = []
        self.save()

    def get_last_conversation(self):
        if self.memory:
            return self.memory[-1]
        return None

    def get_all_conversations(self):
        return self.memory

    def summary(self):
        count = len(self.memory)
        return f"{count} conversation{'s' if count != 1 else ''} stored"

