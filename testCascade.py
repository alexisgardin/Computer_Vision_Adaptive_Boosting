import numpy as np
import os
import cv2 as cv
import argparse

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
tableRect = [(0, 0, 28, 28)]
TRUE_POSITIVE = 0
FALSE_POSITIVE = 1
FALSE_NEGATIVE = 2
VP, VN, FP, FN = 0, 0, 0, 0

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=0, minSize=(20, 20))
    return rects


def draw_rects(img, rects, rank):
    if len(rects) is not 0:
        if rank == TRUE_POSITIVE:
            for x1, y1, x2, y2 in rects:
                cv.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
        if rank == FALSE_POSITIVE:
            for x1, y1, x2, y2 in tableRect:
                cv.rectangle(img, (x1, y1), (x2, y2), RED, 2)
    elif rank == FALSE_NEGATIVE:
        for x1, y1, x2, y2 in tableRect:
            cv.rectangle(img, (x1, y1), (x2, y2), BLUE, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cascade', default="training/cascade.xml")
    parser.add_argument('--labels', default=0)
    args = parser.parse_args()
    cascade = cv.CascadeClassifier(args.cascade)
    data = []
    dataFinal = []
    index = 0
    for i in range(0, 10):
        for filename in os.listdir("test/" + str(i)):
            file_path = "test/" + str(i) + "/" + filename
            print(file_path)
            img = cv.imread(file_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            rects = detect(gray, cascade)
            if i == args.label:
                if len(rects) == 0:
                    draw_rects(img, rects, FALSE_NEGATIVE)
                    FN = FN + 1
                else:
                    draw_rects(img, rects, TRUE_POSITIVE)
                    VP = VP + 1
            else:
                if len(rects) != 0:
                    draw_rects(img, rects, FALSE_POSITIVE)
                    FP = FP + 1
                else:
                    VN = VN + 1
            data.append(img)
            index = index + 1
            if len(data) == 100:
                dataFinal.append(np.concatenate(data, 1))
                data = []
    if len(data) != 0:
        dataFinal.append(np.concatenate(data, 1))
        data = []
    numpy_horizontal_concat = np.concatenate(dataFinal, 0)
    cv.imwrite("result.png", numpy_horizontal_concat)
    sensibilite = VP/(VP+FN)
    print("Sensibilité : " + str(sensibilite))
    print("Spécificité : " + str(VN/(VN+FP)))
    precision = VP/(VP+FP)
    print("Précision : " + str(precision))
    print("F-MESURE : " + str(2*((sensibilite*precision)/(sensibilite+precision))))
    print(numpy_horizontal_concat)
