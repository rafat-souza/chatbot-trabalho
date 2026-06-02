import pandas as pd
import nltk
import string
import unicodedata
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

nltk.download('punkt')
nltk.download('rslp')
nltk.download('stopwords')

gerador = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype=torch.float32
)

def remover_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto)
                   if unicodedata.category(c) != 'Mn')

def preprocess(text):
    text = remover_acentos(str(text).lower())
    tokens = nltk.word_tokenize(text)
    stemmer = nltk.stem.RSLPStemmer()
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')

    return " ".join([stemmer.stem(t) for t in tokens if t not in string.punctuation and t not in stopwords_pt])

try:
    df_base = pd.read_csv("suporte_ti.csv", encoding="utf-8")
    perguntas_originais = df_base['pergunta'].astype(str).tolist()
    respostas_corpus = df_base['resposta'].astype(str).tolist()

    perguntas_proc = [preprocess(doc) for doc in perguntas_originais]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(perguntas_proc)
except Exception as e:
    print(f"Erro ao carregar suporte_ti.csv: {e}")
    perguntas_originais, respostas_corpus, perguntas_proc = [], [], []
    vectorizer, tfidf_matrix = None, None


def get_response(user_input):
    if vectorizer is not None and len(perguntas_proc) > 0:
        user_input_processed = preprocess(user_input)

        user_tfidf = vectorizer.transform([user_input_processed])
        vals = cosine_similarity(user_tfidf, tfidf_matrix)
        melhor_indice = vals.argsort()[0][-1]
        maior_similaridade = vals.flatten()[melhor_indice]

        print(f"Nota de similaridade TF-IDF: {maior_similaridade:.2f}")

        if maior_similaridade > 0.35:
            resposta = respostas_corpus[melhor_indice]
            return f"[Base de Dados Local - Suporte TI]\n\n{resposta}"

    mensagens = [
        {
            "role": "system",
            "content": "Você é um assistente técnico de suporte de TI. A solicitação do usuário não foi encontrada na base de dados interna. Responda o problema relatado de forma técnica, direta e em português."
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
            do_sample=True,
            return_full_text = False
        )

        if isinstance(resultado[0]["generated_text"], list):
            resposta_final = resultado[0]["generated_text"][-1]["content"]
        else:
            resposta_final = resultado[0]["generated_text"]

        return resposta_final.strip()

    except Exception as e:
        return f"Erro interno do modelo: {str(e)}"