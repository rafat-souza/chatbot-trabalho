import nltk
import string
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
    else:
        return pares[questions[index]]

reflexoes = {
    "eu": "você",
    "meu": "seu",
    "você": "eu",
    "seu": "meu",
    "você é": "eu sou",
    "você estava": "eu estava",
    "eu estava": "você estava",
}

print("Descreva o problema do seu computador. Digite 'sair' para finalizar a conversa.")
while True:
    user_query = input("Converse com o bot: ")
    if user_query.lower() == "sair":
        print("ChatBot: Tchau!")
        break
    print(f"ChatBot: {get_response(user_query)}")

