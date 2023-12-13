import cv2
from ultralytics import YOLO
from sort import Sort
import numpy as np
import time
import torch
from utils import *

def video_detection(path_x, alto, ancho):
    video_capture = path_x
    #Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))
    conversion_px_cm = 0.000046

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('best.pt').to(device)
    line = [(100, 2000), (3400, 2000)]

    tracker = Sort()
    start_time = time.time()
    frame_count = 0
    count_obj = 0
    count_id = []

    
    classNames  = ["Maleza", "Plantas de Papa"]
    while True:
        ret, frame = cap.read()
        resultados = model(frame)
        frame = resultados[0].plot(boxes=False)

        for res in resultados:
            clases = res.boxes.cls.cpu().numpy().astype(int)
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            mascaras = res.masks.xy
            trackerMasks, intersection_counter = associate_boxes_with_tracks(boxes, tracker.update(boxes), clases, mascaras, line, count_id)
            trackerMasks = np.array(trackerMasks, dtype=object)
            count_obj += intersection_counter

            for xmin, ymin, xmax, ymax, track_id, clase, mascaras in trackerMasks:
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                track_id = int(track_id)

                cv2.putText(img=frame, text=f"Id:{track_id} Clase: {classNames[clase]}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)
        area = float(alto)*float(ancho)*100
        px_cm = count_obj*conversion_px_cm
        cv2.putText(img=frame, text=f"Area del Terreno: {area} cm2", org=(10, 190), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255), thickness=4)
        cv2.line(frame, line[0], line[1], (255, 255, 0), thickness=2)
        if px_cm >= area/2:
            cv2.putText(img=frame, text="Necesita una limpieza el area", org=(10, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 255), thickness=4)
            cv2.putText(img=frame, text=f"Area Afectada: {px_cm} cm2", org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 255), thickness=4)
        else:
            cv2.putText(img=frame, text=f"Area Afectada: {px_cm} cm2", org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255), thickness=4)
        yield frame

cv2.destroyAllWindows()