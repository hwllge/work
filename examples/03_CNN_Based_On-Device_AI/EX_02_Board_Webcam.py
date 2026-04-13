# 모듈 로딩
import numpy as np
import time
import cv2 

# face detection 초기화
haarFD = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(haarFD))

def processImage(frame):

    # face detection을 위해 색상을 BGR에서 GRAY로 변경
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 영역 좌표 획득
    faces = face_cascade.detectMultiScale(gray)

    # 얼굴 영역 표시
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

# 카메라 설정
cap = cv2.VideoCapture(0) # 0번 카메라 열기
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

# 윈도우 설정
cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', 320+40, 240+60)

startTime = time.time()
while(cap.isOpened()):
    ret,frame=cap.read() # 사진 찍기 -> (240,320,3)
    if not ret: break

    # 이미지 처리
    processImage(frame)

    # FPS 표시
    curTime = time.time()
    fps = 1/(curTime - startTime)
    startTime = curTime
    cv2.putText(frame,f'FPS: {fps:.1f}',(20, 50),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)

    # 이미지 출력
    cv2.imshow('cam',frame)

	 # 10ms 동안 키 입력 대기
    key = cv2.waitKey(10)
    if  key == ord('q'): break

cap.release() # 카메라 닫기
cv2.destroyAllWindows() # 모든 창 닫기
