# 모듈 로딩
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import time
import cv2 
from cvzone.HandTrackingModule import HandDetector
hd = HandDetector(maxHands=1)

# TFLite 모델 로딩
modelPath = 'RPS_MobileNetV2_Augmentation_QAT.tflite'
interpreter = Interpreter(model_path = modelPath) # 모델 로딩
interpreter.allocate_tensors() # tensor 할당
input_details = interpreter.get_input_details() # input tensor 정보 얻기
output_details = interpreter.get_output_details() # output tensor 정보 얻기
input_dtype = input_details[0]['dtype']
input_scale, input_zero = interpreter.get_input_details()[0]['quantization']
output_scale, output_zero = interpreter.get_output_details()[0]['quantization']
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('model input shape:', (height, width))
#print(input_details)
#print(output_details)
print('input_scale = ', input_scale)
print('input_zero = ', input_zero)
print('output_scale = ', output_scale)
print('output_zero = ', output_zero)

ansToText = {0:'scissors', 1:'rock', 2:'paper'}
colorList = [(255,0,0),(0,255,0),(0,0,255)]
IMG_SIZE = 224
offset = 30

def make_square_img(img):
    ho, wo = img.shape[0], img.shape[1]
    aspectRatio = ho/wo
    # 흰색의 background 이미지 준비
    wbg = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    if aspectRatio > 1: # portrait
        k = IMG_SIZE/ho
        wk = int(wo*k)
        img = cv2.resize( img, (wk, IMG_SIZE))
        img_h, img_w = img.shape[0], img.shape[1]
        d = (IMG_SIZE - img_w) // 2
        wbg [:img_h, d:img_w+d] = img
    else: # landscape
        k = IMG_SIZE/wo
        hk = int(ho*k)
        img = cv2.resize( img, (IMG_SIZE, hk))
        img_h, img_w = img.shape[0], img.shape[1]
        d = (IMG_SIZE - img_h) // 2
        wbg [d:img_h+d, :img_w ] = img
    return wbg

def processImage(frame):
    # 손 detect 시도
    hands, _ = hd.findHands(frame, draw=False)
    if not hands: return
    
    # BB(Bounding Box) 얻기
    x, y, w, h = hands[0]['bbox']

    # 범위 초과 확인
    if x<offset or y<offset or x+w+offset>320 or y+h>240: return

    # BB의 좌상단 좌표 구하기 -> 좌와 상 방향으로 offset 만큼 늘림
    x1, y1 = x-offset,  y-offset

    # BB의 우하단 좌표 구하기 -> 우 방향으로만 offset 만큼 늘림
    x2, y2 = x+w+offset, y+h

    # 손만 떼어오기
    img = frame[y1:y2, x1:x2]

	# 떼어온 손을 정사각형으로 만들기
    img = make_square_img(img)

    # BGR을 RGB로 변경
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # 모델의 입력 형태로 수정: (1,224,224,3)
    img = np.expand_dims(img, 0)

    # scale, zero 적용
    img = (img / input_scale) + input_zero

    # 모델에 입력하여 결과 얻기
    #   input tensor 설정
    interpreter.set_tensor(input_details[0]['index'], img.astype(input_dtype))
    #   모델 실행
    interpreter.invoke()
    #   output tensor 얻기
    output_tensor = interpreter.get_tensor(output_details[0]['index'])[0]
    # scale, zero 적용
    predictions = (output_tensor.astype(np.float32) - output_zero) * output_scale
    ans = np.argmax(predictions)
    text = ansToText[ans]

    # BB 표시
    cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[ans], 2)

    # 판정 결과 표시
    cv2.putText(frame,text,(x1,y1-7),cv2.FONT_HERSHEY_PLAIN,2,colorList[ans],2)

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
