# 모듈 로딩
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import time
import cv2 

# LiteRT 모델 로딩
modelPath = 'RPS_MobileNetV2.tflite'
interpreter = Interpreter(model_path = modelPath) # 모델 로딩
interpreter.allocate_tensors() # tensor 할당
input_details = interpreter.get_input_details() # input tensor 정보 얻기
output_details = interpreter.get_output_details() # output tensor 정보 얻기
input_dtype = input_details[0]['dtype']
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('model input shape:', (height, width))
print(input_details)
print(output_details)

ansToText = {0:'scissors', 1:'rock', 2:'paper'}
colorList = [(255,0,0),(0,255,0),(0,0,255)]
IMG_SIZE = 224

def processImage(frame):
    # BGR을 RGB로 변경
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # 모델의 입력 형태로 수정: (1,224,224,3)
    # Normalization 처리 안함 (모델 내에서 -1 ~ 1로 변환 처리함)
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)

    # 모델에 입력하여 결과 얻기
    #   input tensor 설정
    interpreter.set_tensor(input_details[0]['index'], img.astype(input_dtype))
    #   모델 실행
    interpreter.invoke()
    #   output tensor 얻기
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    ans = np.argmax(output_data)
    text = ansToText[ans]

    # 판정 결과 표시
    cv2.putText(frame,text,(180,50),cv2.FONT_HERSHEY_PLAIN,2,colorList[ans],2)

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
