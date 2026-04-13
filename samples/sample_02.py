# 모듈 로딩
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import time
import cv2 

# LiteRT 모델 선택
modelPath = "sample_02.tflite"

# LiteRT 모델 로딩
interpreter = Interpreter(model_path = modelPath) # 모델 로딩
interpreter.allocate_tensors() # tensor 할당

# 모델 정보 얻기 
input_details = interpreter.get_input_details()  # input tensor 정보 얻기
output_details = interpreter.get_output_details() # output tensor 정보 얻기
print(input_details)
print(output_details)
input_index = input_details[0]['index']
output_index = output_details[0]['index']
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']
input_scale, input_zero = interpreter.get_input_details()[0]['quantization']
output_scale, output_zero = interpreter.get_output_details()[0]['quantization']
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('model input shape:', (height, width))

# BB 텍스트 및 색상 정의
ansToText = {0:'scissors', 1:'rock', 2:'paper'}
colorList = [(255,0,0),(0,255,0),(0,0,255)]

# 모델 입력 크기
IMG_SIZE = 320

# Threshold 설정
CONF_TH = 0.5
IOU_TH  = 0.45

def letterbox(img, new_shape=(320,320), color=(114,114,114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)

    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = nw - new_w
    pad_h = nh - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_y, pad_y,
        pad_x, pad_x,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, r, pad_x, pad_y

def nms(boxes, scores, iou_th):
    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1, y1, x2, y2 = boxes.T
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)
        iou = (w*h) / (areas[i] + areas[order[1:]] - w*h + 1e-6)

        order = order[np.where(iou <= iou_th)[0] + 1]

    return keep

def processImage(frame):

    # BGR을 RGB로 변경
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # letterbox 적용
    img_lb, r, pad_x, pad_y = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))

    # 0 ~ 1 사이 값으로 변경
    img = img_lb.astype(np.float32) / 255.0

    # 모델의 입력 형태로 수정: (1,320,320,3)
    img = np.expand_dims(img, axis=0)

    # 입력 타입이 INT8인 경우 처리
    if input_dtype == np.int8:
        img_int8 = (img / input_scale) + input_zero
        img = img_int8.astype(np.int8)

    # 모델에 입력하여 결과 얻기
    #   input tensor 설정
    interpreter.set_tensor(input_index, img)
    #   모델 실행
    interpreter.invoke()
    #   output tensor 얻기: (1,7,2100) -> (7,2100) -> (2100,7)
    raw = interpreter.get_tensor(output_index)[0].transpose()

    # 출력 타입이 INT8인 경우 처리
    if output_dtype == np.int8:
        raw = (raw.astype(np.float32) - output_zero) * output_scale

    # 결과 중 CONF_TH 이상만 모으기
    boxes, scores, class_ids = [], [], []

    for det in raw:
        cx, cy, w, h = det[:4]
        cls_scores = det[4:7]

        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id])
        if score < CONF_TH:
            continue

        cx *= IMG_SIZE; cy *= IMG_SIZE
        w  *= IMG_SIZE; h  *= IMG_SIZE

        x1 = (cx - w/2 - pad_x) / r
        y1 = (cy - h/2 - pad_y) / r
        x2 = (cx + w/2 - pad_x) / r
        y2 = (cy + h/2 - pad_y) / r

        boxes.append([
            int(np.clip(x1,0,frame.shape[1])),
            int(np.clip(y1,0,frame.shape[0])),
            int(np.clip(x2,0,frame.shape[1])),
            int(np.clip(y2,0,frame.shape[0]))
        ])
        scores.append(score)
        class_ids.append(cls_id)

    # NMS 적용
    keep = nms(boxes, scores, IOU_TH)

    # 최종 BB만 그리기
    draw_boxes = [(boxes[i], class_ids[i], scores[i]) for i in keep]

    for (x1,y1,x2,y2), cid, sc in draw_boxes:

        # BB 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[cid], 2)

        # 판정 결과 표시
        cv2.putText(frame, ansToText[cid], (x1,y1-7), cv2.FONT_HERSHEY_PLAIN, 2, colorList[cid], 2)

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
