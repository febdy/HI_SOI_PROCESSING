from pymongo import MongoClient
import datetime
from random import randint

client = MongoClient('mongodb://localhost:27017/')
db = client['hi_soi']

# print(db.collection_names(include_system_collections=False))
# pprint.pprint(db.startup_log.find_one())


def insert_correct_result(video_info, face_move_cnt):
    collection = db['correction_result']

    result = {"user_id": video_info["userId"],
              "video_save_name": video_info["videoSaveName"],
              "face_move_cnt": face_move_cnt,
              "cnt_per_5sec": video_info["cnt_per_5sec"],
              "date": datetime.datetime.now()}

    collection.insert_one(result)
    print("[SUCCESS] Inserted a correction result into MongoDB successfully.")


def get_video_info(video_no):
    collection = db['video_info']
    return collection.find_one({'videoNo': str(video_no)})


# def get_video_info(video_save_name):
#     collection = db['video_info']
#     return collection.find_one({'videoSaveName': video_save_name})


def update_correct_result(video_info):
    collection = db['video_info']
    collection.update_one({'videoNo': video_info['videoNo']},
                          {'$set': {"faceMoveCnt": randint(0, 9),
                                    "miss_location": video_info['miss_location'],
                                    "miss_section": video_info['miss_section'],
                                    "total_video_time": video_info['total_video_time'],
                                    "cnt_per_5sec": video_info["cnt_per_5sec"],
                                    "date": datetime.datetime.now()}})
    #face_move_cnt}})
    print("[SUCCESS] Inserted a correction result into MongoDB successfully.")
