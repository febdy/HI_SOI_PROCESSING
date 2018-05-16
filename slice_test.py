test='from client::::::::::: VideoVo [rn=0, videoNo=26, userId=1235, videoOriginName=ace.mov, videoSaveName=152644638255532698f66-b657-494c-b399-0831a31b9b07.mov, videoExName=.mov, videoPath=adgfdsg.mov, videoSize=639746, videoDate=null, videoThumnail=null, videoCorrectLine=null, videoDelete=null]'

userId_s = test.find('userId')
userId_e = (test[userId_s:]).find(',')
print(test[userId_s + 7: userId_s + userId_e])

videoSaveName_s = test.find('videoSaveName')
videoSaveName_e = (test[videoSaveName_s:]).find(',')
print(test[videoSaveName_s + 14: videoSaveName_s + videoSaveName_e])
