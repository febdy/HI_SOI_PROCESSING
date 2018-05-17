from flask import Flask
from flask_socketio import SocketIO, send
import socket
import conn_pymongo
import keras_tracking

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@socketio.on('message')
def handle_message(message):
    print('received json: ' + message)
    send(1)
    return 2


# @socketio.on('json')
# def handle_message(json):
#     print('received json: ' + str(json))
#     send(1)
#     return 2


@app.route('/')
def start_run_server():
   # run_server()
    return 'Flask running...'


def run_server():
    s = socket.socket()
    host = ''
    port = 5000

    try:
        s.bind((host, port))
    except Exception as error:
        print("Bind Error: ", repr(error))

    s.listen(5)

    conn, addr = s.accept()
    print('got conn from', addr)

#     try:
#         data = conn.recv(1024).decode('utf-8')
#         print("from client:::::::::::", data)
#         conn.sendall(bytes(1))
#         # userId_s = data.find('userId')
#         # userId_e = (data[userId_s:]).find(',')
#         # print(data[userId_s + 7: userId_s + userId_e])
#
#         video_save_name_s = data.find('videoSaveName')  # java에서 넘어온 비디오 정보
#         video_save_name_e = (data[video_save_name_s:]).find(',')
#         video_save_name = data[video_save_name_s + 14: video_save_name_s + video_save_name_e]
#
#         video_info = conn_pymongo.get_video_info(video_save_name)  # MongoDB에서 이름으로 정보 가져오기
# #        result = keras_tracking.do_correction(video_info)
#
# #        conn.sendall(bytes(result))
# #       conn.sendall(bytes(data + "\n", "utf-8"))
#         conn.close()
#     except Exception as error:
#         print("Error:", repr(error))

    s.close()


if __name__ == '__main__':
    socketio.run(app)
