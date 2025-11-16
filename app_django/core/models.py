from django.db import models

from django.db import models

class Chat(models.Model):
    title = models.CharField(max_length=100, blank=True, default="Chat")

    def __str__(self):
        return f"Chat {self.id} - {self.title}"


class Message(models.Model):
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name="messages")
    sender = models.BooleanField()  # True = usuário, False = IA
    text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        sender_name = "Usuário" if self.sender else "IA"
        return f"[{sender_name}] {self.text[:30]}"
