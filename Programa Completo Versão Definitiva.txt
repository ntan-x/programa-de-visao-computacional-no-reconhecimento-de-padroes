import cv2
import numpy as np
import mediapipe as mp
import csv
import datetime
import tkinter as tk
from PIL import Image, ImageTk
import time
from collections import Counter

#referenceFile = "referencia.csv"
#sampleFile = "amostras.csv"
#inserir filepath dos arquivos de Referência e de Amostras
referenceFile = "C:/programacao/python/logs/Reference.csv"
sampleFile = "C:/programacao/python/logs/teste.csv"

#Define número, resolução e enquadramento da webcam
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

#Define resolução da imagem do esqueleto da mão
shapeResolution = 256
shapeOffSet = 25
handSkeletonEmpty = np.zeros((shapeResolution + shapeOffSet*2, shapeResolution + shapeOffSet*2, 3), np.uint8)
handSkeleton = np.zeros((shapeResolution + shapeOffSet*2, shapeResolution + shapeOffSet*2, 3), np.uint8)

handConnections = [[0, 1], [1, 2], [2, 3], [3, 4],
                 [0, 5], [5, 6], [6, 7], [7, 8],
                 [5, 9], [9, 10], [10, 11], [11, 12],
                 [9, 13], [13, 14], [14, 15], [15, 16],
                 [13, 17], [17, 18], [18, 19], [19, 20],
                 [0, 17]]

save = -1
time1 = -1

pTime = 0
cTime = 0
fpsMemory = np.zeros(30)

gestureMemory = [""]*11
previousGesture = ""
Gesture = ""
gestureTimer = 0

subtitleLength = 30
subtitle = [""]*subtitleLength
subtitleIndex = 0
subtitleLock = 0

with open(referenceFile, 'r',  newline='') as csvfile:
    Reference = list(csv.reader(csvfile, delimiter=','))
for row in Reference[:]:
    for i in range(0, 42):
        row[i] = int(row[i])
    row[44] = int(row[44])

class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, trackCon=0.4):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList


detector = handDetector()


def columnVector(matrix, column, rowStart, rowEnd, datatype=int):
    rowIndex = rowEnd-rowStart
    vector = np.empty(rowIndex+1, datatype)
    while rowIndex >= 0:
        vector[rowIndex] = matrix[rowIndex][column]
        rowIndex = rowIndex-1
    return vector


def normalizeVector(vector,inputRange,outputRange):
    index = len(vector) - 1
    result = np.empty(index + 1, dtype=int)
    while (index >= 0):
        result[index] = int(np.interp(vector[index], inputRange, outputRange))
        index = index - 1
    return result


def drawCircleVector(image, xCoordinate, yCoordinate, size = 3, colour = (0, 255, 255), range = (-1, -1), offSet = 10):
    if ((range[0] < 0) or (range[1] < 0) or (range[1] < range[0])):
        index = len(xCoordinate) - 1
        end = 0
    else:
        index = range[1]
        end = range[0]
    while (index >= end):
        cv2.circle(image, (xCoordinate[index] + offSet, yCoordinate[index] + offSet), size, colour, cv2.FILLED)
        index = index - 1
    return


def drawLineVector(image, xCoordinate, yCoordinate, dots, colour = (0, 255, 255), thickness = 2, offSet = 10):
    index = 0
    end = len(dots)
    while (index < end):
        cv2.line(image, (xCoordinate[dots[index][0]] + offSet, yCoordinate[dots[index][0]] + offSet), (xCoordinate[dots[index][1]] + offSet, yCoordinate[dots[index][1]] + offSet), colour, thickness)
        index = index + 1
    return


def identifyGesture(reference, x, y):
    lastLine = len(reference)
    previousDistance = 1000000
    gesture = ""
    for line in range(0, lastLine):
        distance = 0
        row = 0
        while row <= 40:
            distance = distance + ((x[int(row/2)]-int(reference[line][row]))**2+(y[int(row/2)]-int(reference[line][row+1]))**2)
            row += 2
        distance = distance**(1/2)
        if distance < previousDistance:
            previousDistance = distance
            if distance <= reference[line][44]:
                gesture = reference[line][42]
            else:
                gesture = ""
            nearest_gesture = reference[line][42]
            nearest_gesture_tag = reference[line][43]
            nearest_gesture_distance = distance

    return gesture, nearest_gesture, nearest_gesture_tag, nearest_gesture_distance


def shiftVector(vector, shift=1, datatype="float64"):
    length = len(vector)
    shift = int(shift)
    newVector = np.zeros(length, dtype=datatype)
    if shift > 0:
        row = 0
        while row < (length-shift):
            newVector[row + shift] = vector[row]
            row += 1
    elif shift < 0:
        row = length - 1
        while row >= -shift:
            newVector[row + shift] = vector[row]
            row -= 1
    else:
        newVector = np.copy(vector)
    return newVector


def detachXY(reference, gesture=0):
    index = 0
    x = np.empty(21, dtype=int)
    y = np.empty(21, dtype=int)
    while index <= 40:
        x[int(index/2)] = reference[gesture][index]
        y[int(index/2)] = reference[gesture][index+1]
        index += 2
    return x, y


def saveSample(x, y, dType="None", tag="None", file=sampleFile, maxDistance=200, time=datetime.datetime.now()):
    with open(file, "a", newline='') as file:
        data = csv.writer(file, delimiter=",")
        data.writerow([x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3],
                       x[4], y[4], x[5], y[5], x[6], y[6], x[7], y[7],
                       x[8], y[8], x[9], y[9], x[10], y[10], x[11], y[11],
                       x[12], y[12], x[13], y[13], x[14], y[14], x[15], y[15],
                       x[16], y[16], x[17], y[17], x[18], y[18], x[19], y[19],
                       x[20], y[20], dType, tag, maxDistance, time])


def runSaveSample():
    global save, gesture, gestureTag
    if save <= 0:
        save = 1
        gesture = textBox.get()
        gestureTag = textBox2.get()
    gestureTag = textBox2.get()


def runEraseSubtitle():
    global subtitle
    subtitle = [""]*25
    return


displayRefIndex = 0


def runMinus():
    global displayRefIndex
    if displayRefIndex > 0:
        displayRefIndex -= 1
    return


def runPlus():
    global displayRefIndex
    if displayRefIndex < (len(Reference)-1):
        displayRefIndex += 1
    return


#Os modos só servem para adicionar ou retirar algumas informações da tela
mode = 2


def runMode1():
    global mode
    mode = 1
    return


def runMode2():
    global mode
    mode = 2
    return


def runMode3():
    global mode
    mode = 3
    return


#Início da configuração da janela
root = tk.Tk()
root.title("Reconhecimento de Gestos")
root.geometry("954x480")

cameraMain = tk.Label(root)
cameraMain.place(x=0, y=0)
handSkeletonMain = tk.Label(root)
handSkeletonMain.place(x=645, y=0)

buttonM1 = tk.Button(root, text='Modo 1')
buttonM1.config(command=runMode1)
buttonM1.place(x=650, y=360)

buttonM2 = tk.Button(root, text='Modo 2')
buttonM2.config(command=runMode2)
buttonM2.place(x=720, y=360)

buttonM3 = tk.Button(root, text='Modo 3')
buttonM3.config(command=runMode3)
buttonM3.place(x=790, y=360)

minus = tk.Button(root, text=' - ')
minus.config(command=runMinus)
minus.place(x=790, y=330)

plus = tk.Button(root, text=' + ')
plus.config(command=runPlus)
plus.place(x=815, y=330)

eraseSubtitleButton = tk.Button(root, text='Apagar Legenda')
eraseSubtitleButton.config(command=runEraseSubtitle)
eraseSubtitleButton.place(x=650, y=320)

saveSampleButton = tk.Button(root, text='Salvar Amostras')
saveSampleButton.config(command=runSaveSample)
saveSampleButton.place(x=650, y=445)

gesture = tk.StringVar()
textBox = tk.Entry(textvariable=gesture)
textBox.place(x=650, y=420)
text1 = tk.Label(text="Gesto")
text1.place(x=650, y=395)

gestureTag = tk.StringVar()
textBox2 = tk.Entry(textvariable=gestureTag)
textBox2.place(x=780, y=420)
text1 = tk.Label(text="Mão")
text1.place(x=780, y=395)
#Fim da configuração da janela

#Loop principal do programa
while True:
    #Captura de imagem e identificação dos pontos da mão
    success, raw_img = cap.read()
    raw_img = cv2.flip(raw_img, 1)
    if mode == 2:
        camera_img = detector.findHands(raw_img, draw=True)
    else:
        camera_img = detector.findHands(raw_img, draw=False)
    lmList = detector.findPosition(camera_img, draw=False)

    #Cálculo da média do FPS
    cTime = time.time()
    timeDelta = cTime - pTime
    fps = 1 / timeDelta
    pTime = cTime
    fpsMemory = shiftVector(fpsMemory)
    fpsMemory[0] = fps
    fpsMemLen = len(fpsMemory)
    averageFPS = 0
    for index in range(0, fpsMemLen):
        averageFPS = averageFPS + fpsMemory[index]
    averageFPS = averageFPS/fpsMemLen
    cv2.putText(camera_img, f'FPS: {int(averageFPS)}', (0, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    #Rotina quando mão é detectada
    if len(lmList) != 0:
        #Manipulação dos pontos da mão
        xLandmark = columnVector(lmList, 1, 0, 20)
        yLandmark = columnVector(lmList, 2, 0, 20)

        #Calcula enquadramento da mão
        xmax = max(xLandmark)
        ymax = max(yLandmark)
        xmin = min(xLandmark)
        ymin = min(yLandmark)
        dx = xmax - xmin
        dy = ymax - ymin

        #Desenha retângulo vermelho em volta da mão
        cv2.rectangle(camera_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

        #Define a proporção de enquadramento da mão
        if dx > dy:
            xSize = shapeResolution
            ySize = int((float(dy)/(dx))*shapeResolution)
        elif dx < dy:
            ySize = shapeResolution
            xSize = int((float(dx)/(dy))*shapeResolution)
        else:
            xSize = shapeResolution
            ySize = shapeResolution

        #Normaliza os pontos da mão
        px = normalizeVector(xLandmark, [xmin, xmax], [0, xSize])
        py = normalizeVector(yLandmark, [ymin, ymax], [0, ySize])

        #Rotina para diminuir oscilação do gesto mostrado
        gestureMemory = shiftVector(gestureMemory, datatype="str")
        gestureMemory[0], nearestGesture, nearestGestureTag, nearestGestureDistance = identifyGesture(Reference, px, py)
        cnt = Counter(gestureMemory)
        cnt_gesture = sorted(cnt, key=cnt.get, reverse=True)
        cnt_sorted = [[0] * 2 for _ in range(len(cnt_gesture))]
        aux = 0
        for r in cnt_gesture:
            cnt_sorted[aux][0] = r
            cnt_sorted[aux][1] = cnt[r]
            aux += 1
        previousGesture = Gesture
        Gesture = cnt_sorted[0][0]
        cv2.putText(camera_img, f'Gesto: {Gesture}', (250, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        #Rotina da legenda
        gestureLockTime = 3
        if (previousGesture == Gesture) and (gestureTimer < gestureLockTime):
            gestureTimer = gestureTimer + timeDelta
        if (previousGesture != Gesture) or (Gesture == ""):
            gestureTimer = 0
            subtitleLock = 0
        if gestureTimer > gestureLockTime:
            gestureTimer = gestureLockTime
        if (subtitleLock == 0) and (gestureTimer >= gestureLockTime):
            subtitle[subtitleIndex] = Gesture
            subtitleIndex += 1
            subtitleLock = 1
        if subtitleIndex >= len(subtitle):
            subtitle = shiftVector(subtitle, -1, "str")
            subtitleIndex -= 1
        if gestureTimer > 0:
            cv2.circle(camera_img, (600, 40), (int(30 * (gestureTimer / gestureLockTime))), (0, 0, 255), -1)
        cv2.circle(camera_img, (600, 40), 30, (0, 0, 255), 2)

        #Não desenha esqueleto da mão com os pontos normalizados no modo 1
        if mode == 1:
            handSkeleton = np.copy(handSkeletonEmpty)
        #Exibe o gesto mais próximo, a tag e a distância quando no modo 2 ou 3
        else:
            cv2.putText(camera_img, f'Gesto mais proximo: {nearestGesture} {nearestGestureTag}  |  Distancia={int(nearestGestureDistance)}', (5, 425), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)

        #Desenha o esqueleto da mão com os pontos normalizados no modo 2
        if mode == 2:
            handSkeleton = np.copy(handSkeletonEmpty)
            drawCircleVector(handSkeleton, px, py, 5, (255, 255, 255), offSet=shapeOffSet)
            drawLineVector(handSkeleton, px, py, handConnections, (255, 255, 255), 2, offSet=shapeOffSet)

        #Rotina para salvar as amostras
        #prepTime é o tempo em segundos antes de começar a salvar
        #captureTime é o tempo em segundos que o programa fica salvando
        prepTime = 5
        captureTime = 5
        if save == 1:
            time2 = time.time()
            if time1 < 0:
                time1 = time.time()
            saveTime = time2-time1
            if saveTime < prepTime:
                cv2.putText(camera_img, f'Prepare-se: {int(prepTime-saveTime)}', (5, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            else:
                gestureTagStr = str(gestureTag)
                gestureStr = str(gesture)
                saveSample(px, py, gesture, gestureTag)
                cv2.putText(camera_img, f'Salvando: {int((prepTime+captureTime)-saveTime)}', (5, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            if saveTime > (prepTime+captureTime):
                save = -1
                time1 = -1

    #Desenha a legenda na tela
    cv2.rectangle(camera_img, (0, 435), (640, 480), (0, 0, 0), -1)
    cv2.putText(camera_img, f'{"".join(subtitle)}', (20, 465), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    #Quando no modo 3, desenha o esqueleto da mão dos gestos da matriz Referência
    if mode == 3:
        handSkeleton = np.copy(handSkeletonEmpty)
        pxRef, pyRef = detachXY(Reference, displayRefIndex)
        drawCircleVector(handSkeleton, pxRef, pyRef, 5, (255, 255, 255), offSet=shapeOffSet)
        drawLineVector(handSkeleton, pxRef, pyRef, handConnections, (255, 255, 255), 2, offSet=shapeOffSet)
        cv2.putText(handSkeleton, f'{Reference[displayRefIndex][42]}  {Reference[displayRefIndex][43]}', (10, 295), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    #Exibe a imagem da câmera na janela
    cv2image = cv2.cvtColor(camera_img, cv2.COLOR_BGR2RGBA)
    imgTemp = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=imgTemp)
    cameraMain.imgtk = imgtk
    cameraMain.configure(image=imgtk)

    #Exibe a imagem do esqueleto da mão na janela
    cv2image = cv2.cvtColor(handSkeleton, cv2.COLOR_BGR2RGBA)
    imgTemp = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=imgTemp)
    handSkeletonMain.imgtk = imgtk
    handSkeletonMain.configure(image=imgtk)

    #Atualiza tudo que é exibido na janela (root)
    root.update()

    cv2.waitKey(1)

