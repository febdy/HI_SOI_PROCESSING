from flask import Flask
from flask_socketio import SocketIO, send, emit
import conn_pymongo
import read_video

app = Flask(__name__)
app.config.from_object('app_config')
socketio = SocketIO(app)


@socketio.on('connect')
def conn_connect():
    print('Client connected.')


@socketio.on('disconnect')
def conn_disconnect():
    print('Client disconnected.')


# js로부터 video_no을 받아 처리 시작
@socketio.on('video_data')
def handle_message(video_no):
    print('received videoNo: ' + str(video_no))

    video_info = get_video_info(video_no)  # video_no으로 mongo에서 비디오 정보 가져옴
    result = read_video.do_process(video_info)  # process 시작

    if result is not 0:  # process 처리가 모두 제대로 끝나면
        socketio.send(video_no)  # video_no을 다시 js에 보냄
    else:
        socketio.send(0)

    print("sent message to client.")


# video_no을 이용해 mongo에서 비디오 정보 가져옴
def get_video_info(video_no):
    try:
        video_info = conn_pymongo.get_video_info(video_no)
    except Exception as e:
        print("get_video_info error :::: ", repr(e))

    return video_info


if __name__ == '__main__':
    socketio.run(app)
