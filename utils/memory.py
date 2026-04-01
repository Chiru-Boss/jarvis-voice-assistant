class ConversationMemory:
    def __init__(self):
        self.memory = []

    def add_conversation(self, user_input, bot_response):
        self.memory.append({'user_input': user_input, 'bot_response': bot_response})

    def get_memory(self):
        return self.memory

    def clear_memory(self):
        self.memory = []

    def get_last_conversation(self):
        if self.memory:
            return self.memory[-1]
        return None

    def get_all_conversations(self):
        return self.memory
