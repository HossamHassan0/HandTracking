import cv2
import mediapipe
import time

cap = cv2.VideoCapture(0)

mphands = mediapipe.solutions.hands   #detect about 21 landmark in the hand
hands = mphands.Hands()
mpdraw = mediapipe.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    frame, img = cap.read()
    results = hands.process(img)
    print(results)
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            #for idd, lm in enumerate(handlms.landmark):
                #print(idd, lm)
                #h, w, c = img.shape
                #cx, cy = int(lm, x*w), int(lm, y*h)
                #print(idd, cx, cy)
                #if idd == 0:
                    #cv2.circle(img, (cx, cy), 9, (255, 0, 255), cv2.FILLED)

            mpdraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)

    cTime = time.time()
    fbs = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fbs)), (15, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), cv2.LINE_AA)

    cv2.imshow("Frame", img)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cv2.release()
