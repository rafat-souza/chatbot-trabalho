import nltk
import string
import random
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pares_brutos = {

    # conversa casual
    ("oi", "olá", "opa", "e aí"): "Olá! Sou seu assistente técnico. Como posso ajudar?",
    ("quem é você", "o que você faz"): "Sou um bot de suporte para problemas de computador.",

    # suporte
    "computador não liga": "Verifique se o cabo de força está conectado à tomada e ao gabinete.",
    "internet lenta": "Reinicie o seu roteador e verifique se há muitos dispositivos conectados.",
    "tela azul": "Geralmente indica erro de hardware ou driver. Tente reiniciar em modo de segurança.",
    "computador travando": "Pressione Ctrl+Alt+Del e feche programas que consomem muita memória.",
    "monitor sem imagem": "Verifique se o cabo de vídeo (HDMI/VGA) está bem encaixado na placa de vídeo.",
    "áudio não funciona": "Verifique se o driver de som está atualizado e se a saída correta está selecionada."
}

pares = {}
for chave, resposta in pares_brutos.items():
    if isinstance(chave, tuple):
        for sub_chave in chave:
            pares[sub_chave] = resposta
    else:
        pares[chave] = resposta

def preprocess(text):
    text = remover_acentos(text.lower())
    tokens = nltk.word_tokenize(text)
    stemmer = nltk.stem.RSLPStemmer()
    return " ".join([stemmer.stem(t) for t in tokens if t not in string.punctuation])

def remover_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto)
                   if unicodedata.category(c) != 'Mn')

saudacoes = ["oi", "olá", "opa", "e aí", "quem é você", "o que você faz"]
saudacoes_proc = [preprocess(s) for s in saudacoes]

def get_response(user_input):
    user_input_processed = preprocess(user_input)

    questions = list(pares.keys())
    questions_processed = [preprocess(q) for q in questions]

    questions_processed.append(user_input_processed)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(questions_processed)

    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = vals.argsort()[0][-1]

    flat = vals.flatten()
    flat.sort()

    if flat[-1] < 0.3:
        return "Desculpe, não entendi o problema. Pode detalhar melhor?"

    resposta_base = pares[questions[index]]
    pergunta_encontrada_proc = questions_processed[index]

    if pergunta_encontrada_proc in saudacoes_proc:
        return resposta_base
    else:
        sugestoes = [
            "Posso ajudar com algo mais?",
            "Tem mais alguma dúvida técnica?",
            "Algo mais está se comportando de forma estranha no PC?"
        ]
        return f"{resposta_base}\n\nChatBot: {random.choice(sugestoes)}"
