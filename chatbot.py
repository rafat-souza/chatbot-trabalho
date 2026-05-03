import nltk
import string
import unicodedata
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

gerador = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

base_dados = [
    "Computador não liga: Verificar conexão do cabo de energia, chave da fonte de alimentação e conexões do painel frontal da placa-mãe.",
    "Tela Azul da Morte (BSOD): Isolar falhas de memória RAM com o software Memtest86, verificar temperaturas e atualizar drivers críticos.",
    "Lentidão do sistema: Monitorar o Gerenciador de Tarefas para identificar gargalos de CPU/Disco/RAM, desativar programas de inicialização desnecessários e verificar a integridade do SSD/HD.",
    "Sem conexão com a internet: Reiniciar o modem/roteador, verificar o cabo de rede (RJ45), ou redefinir as configurações de rede (flush DNS).",
    "Superaquecimento ou desligamento repentino: Limpar poeira dos dissipadores, substituir a pasta térmica do processador e garantir que os fans (ventoinhas) estejam operando adequadamente.",
    "Sem saída de áudio: Checar o dispositivo de reprodução padrão no Windows, reinstalar os drivers de áudio da placa-mãe e testar conectores frontais e traseiros.",
    "Monitor sem vídeo: Verificar cabo HDMI/DisplayPort, limpar contatos da memória RAM com borracha e testar a conexão diretamente na placa-mãe caso a placa de vídeo dedicada falhe."
]


def preprocess(text):
    text = remover_acentos(text.lower())
    tokens = nltk.word_tokenize(text)
    stemmer = nltk.stem.RSLPStemmer()
    return " ".join([stemmer.stem(t) for t in tokens if t not in string.punctuation])


def remover_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto)
                   if unicodedata.category(c) != 'Mn')

base_dados_proc = [preprocess(doc) for doc in base_dados]

def get_response(user_input):
    user_input_processed = preprocess(user_input)
    documentos = base_dados_proc + [user_input_processed]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documentos)

    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = vals.argsort()[0][-1]
    similaridade = vals.flatten()[index]

    if similaridade > 0.1:
        contexto = base_dados[index]
    else:
        contexto = "Não há informações específicas na base de dados. Forneça instruções gerais e peça mais detalhes do problema."

    mensagens = [
        {
            "role": "system",
            "content": "Você é um assistente técnico de suporte. Seja direto nas respostas. Utilize a informação de contexto para embasar a solução. Não utilize tom de brincadeira."
        },
        {
            "role": "user",
            "content": f"Contexto recuperado da base: {contexto}\n\nProblema relatado pelo usuário: {user_input}"
        }
    ]

    try:
        resultado = gerador(
            mensagens,
            max_new_tokens=150,
            temperature=0.2,
            do_sample=True
        )

        resposta_final = resultado[0]["generated_text"][-1]["content"]
        return resposta_final.strip()

    except Exception as e:
        return f"Erro interno do modelo: {str(e)}"