import random
import nltk
from nltk.chat.util import Chat, reflections

pares = [
    [
        r"Oi|Olá|E aí",
        ["Olá! Como posso te ajudar?", "Olá! Qual a sua dúvida sobre computadores?"],
    ],
    [
        r"Qual é o seu nome?",
        ["Meu nome é ChatBot.", "Você pode me chamar de chatbot.", "Sou o chatbot."],
    ],
    [
        r"(.*)/?",
        ["Desculpe, ainda não tenho uma resposta para essa pergunta", "Pode reformular a pergunta?"],
    ],
]

pares.extend([
    [r"(.*)", ["Entendi. Diga-me mais.", "Pode me contar mais sobre isso?"]],
])

reflexoes = {
    "eu": "você",
    "meu": "seu",
    "você": "eu",
    "seu": "meu",
    "você é": "eu sou",
    "você estava": "eu estava",
    "eu estava": "você estava",
}

chatbot = Chat(pares, reflections)

print("Olá! Eu sou o helpbot. Digite 'sair' para encerrar a conversa.")

while True:
    user_input = input("Converse com o bot: ")
    if user_input.lower() == "sair":
        print("ChatBot: Adeus!")
        break
    response = chatbot.respond(user_input)
    print("ChatBot: ", response)

