import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from time import time

########################
offsetPercentageW = 10
offsetPercentageH = 20
confidence = 0.8
camWidth, camHeight = 640, 480
floatingPoint = 6
Save = True
blurThreshold = 35  # Larger is more focus
outputFolderPath = r"G:\Ahmed\Ai Projects\Ai keyboard\Dataset\DataCollect"
debug = False
classID = 0  # 0 is fake and 1 is real

########################


cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = FaceDetector()

while True:
    success, img = cap.read()
    imgOut = img.copy()

    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  # True False values indicating if the faces are blur or not
    listInfo = []  # The normalized values and the class name for the label txt file
    if bboxs:
        for bbox in bboxs:
            x, y, h, w = bbox["bbox"]
            score = bbox["score"][0]
            print(x, y, h, w)
            # ------  Check the score --------
            if score > confidence:

                # ---------- Adding Offset to the Detected Face ---------
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                offsetH = (offsetPercentageW / 100) * h
                y = int(y - offsetH * 4)
                h = int(h + offsetH * 5)

                # ------  To avoid values below 0 --------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # ---------- Find Blurriness -----------------------------

                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurrValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurrValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)
                # ------  Normalize Values  --------
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2

                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                # print(xcn, ycn, wn, hn)

                # ------  To avoid values above 1 --------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1
                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ---------- Drawing -----------------------------

                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% blurr :{blurrValue}', (x, y + 20), scale=2)
                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% blurr :{blurrValue}', (x, y + 20), scale=2)

            # ------  To Save --------
            if Save:
                if all(listBlur) and listBlur != []:
                    # ------  Save Image  --------
                    timeNow = time()
                    timeNow = str(timeNow).split('.')
                    timeNow = timeNow[0] + timeNow[1]
                    cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                    # ------  Save Label Text File  --------
                    for info in listInfo:
                        f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                        f.write(info)
                        f.close()
        cv2.imshow("Imafe", imgOut)
        cv2.waitKey(1)
