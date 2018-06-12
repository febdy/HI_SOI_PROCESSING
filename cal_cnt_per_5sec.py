# 5초 단위로 움직이는 횟수 구하기
def check_cnt_per_5sec(cnt_per_5sec, frame_cnt, fps):
    sec = round(int(frame_cnt / fps))
    print("chk_cnt_per_5sec", sec, len(cnt_per_5sec))
    i = 0

    if sec != 0 and sec % 5 == 0:
        i = (sec // 5) - 1
    else:
        i = sec // 5

    while len(cnt_per_5sec) <= i:
        cnt_per_5sec.append(0)

    cnt_per_5sec[i] += 1

    return cnt_per_5sec