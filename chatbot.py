import requests
import html
import re
import nltk
import string
import unicodedata
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

gerador = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype=torch.float16,
)

# 2. Funções de Pré-processamento NLTK
def remover_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto)
                   if unicodedata.category(c) != 'Mn')

def preprocess(text):
    text = remover_acentos(str(text).lower())
    tokens = nltk.word_tokenize(text)
    stemmer = nltk.stem.RSLPStemmer()
    return " ".join([stemmer.stem(t) for t in tokens if t not in string.punctuation])

def buscar_stack_exchange(query, site="pt.stackoverflow"):
    url = "https://api.stackexchange.com/2.3/search/advanced"
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": site,
        "accepted": "True",
        "filter": "withbody"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("items", [])
    except Exception as e:
        print(f"Erro ao acessar API: {e}")
        return []


def extrair_resposta_aceita(answer_id, site="pt.stackoverflow"):
    url = f"https://api.stackexchange.com/2.3/answers/{answer_id}"
    params = {"site": site, "filter": "withbody"}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("items"):
                corpo_html = html.unescape(data["items"][0]["body"])
                texto_limpo = re.sub(r'<[^>]+>', '', corpo_html)
                return texto_limpo.strip()
    except Exception:
        pass
    return None


def get_response(user_input):
    itens_se = buscar_stack_exchange(user_input)

    if itens_se:
        perguntas_se = [html.unescape(item["title"]) for item in itens_se]

        user_input_processed = preprocess(user_input)
        perguntas_proc = [preprocess(q) for q in perguntas_se]

        documentos = perguntas_proc + [user_input_processed]
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(documentos)

        vals = cosine_similarity(tfidf[-1], tfidf[:-1])
        melhor_indice = vals.argsort()[0][-1]
        maior_similaridade = vals.flatten()[melhor_indice]

        if maior_similaridade > 0.15:
            resposta_aceita_id = itens_se[melhor_indice].get("accepted_answer_id")
            if resposta_aceita_id:
                texto_resposta = extrair_resposta_aceita(resposta_aceita_id)
                if texto_resposta:
                    return f"[Base de Conhecimento Stack Exchange - NLTK/TF-IDF]\n\n{texto_resposta}"

    mensagens = [
        {
            "role": "system",
            "content": "Você é um assistente técnico de suporte de TI. A solicitação do usuário não foi encontrada na base de dados externa. Responda o problema relatado de forma técnica, direta e em português."
        },
        {
            "role": "user",
            "content": f"Problema relatado: {user_input}"
        }
    ]

    try:
        resultado = gerador(
            mensagens,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True
        )
        return resultado[0]["generated_text"][-1]["content"].strip()

    except Exception as e:
        return f"Erro interno do modelo: {str(e)}"