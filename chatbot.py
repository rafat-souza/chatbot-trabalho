import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('rslp')

pares = {

    # conversa casual
    ("oi", "olá", "opa", "e aí"): "Olá! Sou seu assistente técnico. Como posso ajudar?",
    ("quem é você", "o que você faz"): "Sou um bot de suporte para problemas de computador.",

    # suporte
    "computador não liga": "Verifique se o cabo de força está conectado à tomada e ao gabinete.",
    "internet lenta": "Reinicie o seu roteador e verifique se há muitos dispositivos conectados.",
    "tela azul": "Geralmente indica erro de hardware ou driver. Tente reiniciar em modo de segurança.",
    "computador travando": "Pressione Ctrl+Alt+Del e feche programas que consomem muita memória.",
    "monitor sem imagem": "Verifique se o cabo de vídeo (HDMI/VGA) está bem encaixado na placa de vídeo.",
    "audio nao funciona": "Verifique se o driver de som está atualizado e se a saída correta está selecionada."
}


reflexoes = {
    "eu": "você",
    "meu": "seu",
    "você": "eu",
    "seu": "meu",
    "você é": "eu sou",
    "você estava": "eu estava",
    "eu estava": "você estava",
}

print("Olá! Eu sou o helpbot. Digite 'sair' para encerrar a conversa.")

while True:
    user_input = input("Converse com o bot: ")
    if user_input.lower() == "sair":
        print("ChatBot: Adeus!")
        break
    response = chatbot.respond(user_input)
    print("ChatBot: ", response)

