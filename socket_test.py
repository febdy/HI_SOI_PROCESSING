import socket
import conn_pymongo
import keras_tracking

def run_server():
    s = socket.socket()
    host = ''
    port = 9999

    try:
        s.bind((host, port))
    except:
        print("Bind Error.")

    s.listen(5)

    while True:
        conn, addr = s.accept()
        print('got conn from', addr)

        try:
            data = conn.recv(1024).decode('utf-8')
            print("from client:::::::::::", data)

            # userId_s = data.find('userId')
            # userId_e = (data[userId_s:]).find(',')
            # print(data[userId_s + 7: userId_s + userId_e])

            video_save_name_s = data.find('videoSaveName')  # java에서 넘어온 비디오 정보
            video_save_name_e = (data[video_save_name_s:]).find(',')
            video_save_name = data[video_save_name_s + 14: video_save_name_s + video_save_name_e]

            video_info = conn_pymongo.get_video_info(video_save_name)  # MongoDB에서 이름으로 정보 가져오기
            keras_tracking.do_correction(video_info)

    #        conn.sendall(data)
    #        conn.sendall(bytes(data + "\n", "utf-8"))
            conn.close()
        except:
            print("Data doesn't exist")

    s.close()


if __name__ == '__main__':
    run_server()
