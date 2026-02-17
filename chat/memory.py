class ChatMemory:
    def __init__(self, max_turns=5):
        self.history = []
        self.max_turns = max_turns

    def add_user_message(self, message):
        self.history.append(("user", message))
        self._trim()

    def add_assistant_message(self, message):
        self.history.append(("assistant", message))
        self._trim()

    def get_formatted_history(self):
        formatted = ""
        for role, msg in self.history:
            formatted += f"{role.capitalize()}: {msg}\n"
        return formatted

    def _trim(self):
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2 :]
