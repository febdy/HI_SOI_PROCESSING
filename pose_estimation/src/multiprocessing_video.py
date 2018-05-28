from multiprocessing import Process, Pool, TimeoutError
import time
import os

def f(x):
    return x*x

if __name__ == '__main__':
    pool = Pool(processes=4)              # start 4 worker processes

    # print "[0, 1, 4,..., 81]"
    print(pool.map(f, range(10)))

    # print same numbers in arbitrary order
    for i in pool.imap_unordered(f, range(10)):
        print(i)

    # evaluate "f(20)" asynchronously
    res = pool.apply_async(f, (20,))      # runs in *only* one process
    print(res.get(timeout=1))            # prints "400"

    # evaluate "os.getpid()" asynchronously
    res = pool.apply_async(os.getpid, ()) # runs in *only* one process
    print(res.get(timeout=1))              # prints the PID of that process

    # launching multiple evaluations asynchronously *may* use more processes
    multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
    print([res.get(timeout=1) for res in multiple_results])

    # make a single worker sleep for 10 secs
    res = pool.apply_async(time.sleep, (10,))
    try:
        print(res.get(timeout=1))
    except TimeoutError:
        print("We lacked patience and got a multiprocessing.TimeoutError")

# # parser.py
# import requests
# from bs4 import BeautifulSoup as bs
# import time
#
# from multiprocessing import Pool # Pool import하기
#
#
# def get_links(): # 블로그의 게시글 링크들을 가져옵니다.
#     req = requests.get('https://beomi.github.io/beomi.github.io_old/')
#     html = req.text
#     soup = bs(html, 'html.parser')
#     my_titles = soup.select(
#         'h3 > a'
#         )
#     data = []
#
#     for title in my_titles:
#         data.append(title.get('href'))
#     return data
#
# def get_content(link):
#     abs_link = 'https://beomi.github.io'+link
#     req = requests.get(abs_link)
#     html = req.text
#     soup = bs(html, 'html.parser')
#     # 가져온 데이터로 뭔가 할 수 있겠죠?
#     # 하지만 일단 여기서는 시간만 확인해봅시다.
#     print(soup.select('h1')[0].text) # 첫 h1 태그를 봅시다.
#
# if __name__=='__main__':
#     start_time = time.time()
#     pool = Pool(processes=4) # 4개의 프로세스를 사용합니다.
#     pool.map(get_content, get_links()) # get_contetn 함수를 넣어줍시다.
#     print("--- %s seconds ---" % (time.time() - start_time))