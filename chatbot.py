import pandas as pd
import nltk
import string
import unicodedata
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

nltk.download('punkt', quiet=True)
nltk.download('rslp', quiet=True)
nltk.download('stopwords', quiet=True)

classificador_intencao = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    device_map="cpu"
)

gerador = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device_map="cpu",
    dtype=torch.float32
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
    df_base = pd.read_csv("base_conhecimento_ti.csv", encoding="utf-8")
    topicos_corpus = df_base['topico'].astype(str).tolist()
    conteudos_corpus = df_base['conteudo'].astype(str).tolist()

    conteudos_proc = [preprocess(doc) for doc in conteudos_corpus]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(conteudos_proc)
except Exception as e:
    print(f"Erro ao carregar base_conhecimento_ti.csv: {e}")
    topicos_corpus, conteudos_corpus, conteudos_proc = [], [], []
    vectorizer, tfidf_matrix = None, None

def get_response(user_input):
    labels_permitidos = ["suporte técnico de ti", "problema de computador ou rede", "outros assuntos"]
    resultado_zero_shot = classificador_intencao(user_input, candidate_labels=labels_permitidos)

    melhor_label = resultado_zero_shot["labels"][0]
    score = resultado_zero_shot["scores"][0]

    print(f"Intenção detectada: {melhor_label} (Score: {score:.2f})")

    if melhor_label == "outros assuntos" or score < 0.40:
        return "Sou um assistente exclusivo de TI e não respondo sobre outros assuntos."

    contexto_recuperado = ""
    prefixo_frontend = ""

    if vectorizer is not None and len(conteudos_proc) > 0:
        user_input_processed = preprocess(user_input)
        user_tfidf = vectorizer.transform([user_input_processed])
        vals = cosine_similarity(user_tfidf, tfidf_matrix)
        melhor_indice = vals.argsort()[0][-1]
        maior_similaridade = vals.flatten()[melhor_indice]

        print(f"Nota de similaridade TF-IDF: {maior_similaridade:.2f}")

        if maior_similaridade > 0.25:
            contexto_recuperado = conteudos_corpus[melhor_indice]
            print(f"-> Base local acionada. Tópico: {topicos_corpus[melhor_indice]}")
            prefixo_frontend = "[Base de Dados Local - RAG]\n\n"
        else:
            print("-> Contexto não encontrado. Utilizando conhecimento do LLM (Fallback).")
            prefixo_frontend = "[Conhecimento do LLM - Fallback]\n\n"

    if contexto_recuperado:
        prompt_system = f"Você é um chatbot e seu nome é Bob. Não temos email, a única forma de comunicação com o usuário é o seu chat. Sua tarefa é explicar a solução para o problema relatado utilizando APENAS este texto:\n\n{contexto_recuperado}"
    else:
        prompt_system = "Você é um técnico de TI. Dê uma solução técnica e direta para o problema relatado."

    mensagens = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": f"Problema relatado: {user_input}"}
    ]

    try:
        resultado = gerador(
            mensagens,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.15,
            return_full_text=False
        )

        if isinstance(resultado[0]["generated_text"], list):
            resposta_final = resultado[0]["generated_text"][-1]["content"]
        else:
            resposta_final = resultado[0]["generated_text"]

        return prefixo_frontend + resposta_final.strip()

    except Exception as e:
        return f"Erro interno do modelo: {str(e)}"