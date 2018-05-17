from pymongo import MongoClient
import datetime

client = MongoClient('mongodb://localhost:27017/')
db = client['hi_soi']

# print(db.collection_names(include_system_collections=False))
# pprint.pprint(db.startup_log.find_one())


def insert_correct_result(video_info, face_move_cnt):
    collection = db['correction_result']

    result = {"user_id": video_info["userId"],
              "video_save_name": video_info["videoSaveName"],
              "face_move_cnt": face_move_cnt,
              "date": datetime.datetime.now()}

    collection.insert_one(result)
    print("Insert correction result to MongoDB success.")


def get_video_info(video_save_name):
    collection = db['video_info']
    return collection.find_one({'videoSaveName': video_save_name})
