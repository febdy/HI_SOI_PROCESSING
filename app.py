from flask import Flask
from flask_socketio import SocketIO, send, emit
import conn_pymongo
import read_video

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@socketio.on('connect')
def conn_connect():
    print('Client connected.')


@socketio.on('disconnect')
def conn_disconnect():
    print('Client disconnected.')


@socketio.on('video_data')
def handle_message(video_no):
    print('received videoNo: ' + str(video_no))

    video_info = get_video_info(video_no)
    result = read_video.read_video(video_info)
    socketio.emit('result', result)


def get_video_info(video_no):
    try:
        video_info = conn_pymongo.get_video_info(video_no)
    except Exception as e:
        print("get_video_info error :::: ", repr(e))

    return video_info


if __name__ == '__main__':
    socketio.run(app)
