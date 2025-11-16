from django.shortcuts import render, redirect, get_object_or_404
from .models import Chat, Message
from .ia_model import generate_response

def chat_view(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id)

    if request.method == "POST":
        text = request.POST.get("text")

        # Salva mensagem do usuário
        Message.objects.create(
            chat=chat,
            sender=True,
            text=text
        )

        # Resposta temporária da IA
        Message.objects.create(
            chat=chat,
            sender=False,
            text="Resposta da IA (placeholder)."
        )

        return redirect("chat_view", chat_id=chat_id)

    messages = chat.messages.order_by("timestamp")
    all_chats = Chat.objects.all().order_by("id")

    return render(
        request,
        "chat.html",
        {
            "chat": chat,
            "messages": messages,
            "all_chats": all_chats,
        }
    )

def new_chat(request):
    chat = Chat.objects.create()
    return redirect("chat_view", chat_id=chat.id)

def send_message(request, chat_id):
    if request.method == "POST":
        text = request.POST.get("text")
        chat = get_object_or_404(Chat, id=chat_id)

        # 1) salva mensagem do usuário no banco de dados
        Message.objects.create(
            chat=chat,
            sender=True,  # True = usuário
            text=text
        )

        # 2) Chama o modelo pytorch e gera a resposta
        ai_response = generate_response(text)

        
        # 3) Salva a resposta da IA no banco de dados
        Message.objects.create(
            chat=chat,
            sender=False,  # False = IA
            text=ai_response
        )

    return redirect("chat_view", chat_id=chat_id)