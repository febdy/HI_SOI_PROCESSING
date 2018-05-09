import socket

from flask import Flask

app = Flask(__name__)


@app.route('/')
def start_flask():
    start_socket()

    return 'Flask_Running'


def start_socket():
    host = ''
    port = '5000'
    with socket.socket() as s:
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        msg = conn.recv(1024)
        conn.sendall(msg)
        conn.clse()
