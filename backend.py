from flask import Flask, request, jsonify
from flask_cors import CORS

from chatbot import get_response

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    dados = request.get_json()

    if not dados or 'mensagem' not in dados:
        return jsonify({'erro': 'Nenhuma mensagem recebida.'}), 400

    mensagem_usuario = dados['mensagem']

    resposta_bot = get_response(mensagem_usuario)

    return jsonify({'resposta': resposta_bot})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)