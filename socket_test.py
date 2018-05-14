import socket


def run_server():
    s = socket.socket()
    host = ''
    port = 9998
    s.bind((host, port))
    s.listen(5)
    conn = None

    while True:
        if conn is None:
            print("waiting for conn..")
            conn, addr = s.accept()
            print('got conn from', addr)
        else:
            print()
            print("waiting for response..")
            data = conn.recv(1024)

            if not data:
                break

            print("from client:::::::::::", data.decode('utf-8'))
            conn.sendall(data)
            print(conn)

    s.close()


if __name__ == '__main__':
    run_server()
