import requests

url = 'http://127.0.0.1:5000/api/chat'
dados = {'mensagem': 'está dando tela azul'}

resposta = requests.post(url, json=dados)
print(resposta.json())