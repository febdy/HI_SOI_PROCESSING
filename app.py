from flask import Flask
from flask_socketio import SocketIO, send
import conn_pymongo
import keras_tracking

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

    result = get_video_info(video_no)
    send(result)


def get_video_info(video_no):
    result = 0

    try:
        video_info = conn_pymongo.get_video_info(video_no)
        result = keras_tracking.do_correction(video_info)
    except Exception as e:
        print("get_video_info error :::: ", repr(e))

    return result


if __name__ == '__main__':
    socketio.run(app)
