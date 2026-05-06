import cv2
import time
import numpy as np
import mediapipe as mp
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
wCam, hCam = 640, 480       # Camera width and height
camId = 0                   # 0 is usually the default laptop webcam (Try 1 if 0 fails)

# ----------------------------------------------------------------
# SETUP MEDIA PIPE (HAND TRACKING)
# ----------------------------------------------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,        # Track only 1 hand
    min_detection_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# ----------------------------------------------------------------
# SETUP VOLUME CONTROL (FIXED FOR 2025)
# ----------------------------------------------------------------
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.EndpointVolume # <--- THIS IS THE FIX
    volume = interface
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
except Exception as e:
    print(f"Volume Error: {e}")
    minVol = -65
    maxVol = 0

vol = 0
volBar = 400
volPer = 0

# ----------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------
cap = cv2.VideoCapture(camId)
cap.set(3, wCam)
cap.set(4, hCam)

print("System Starting... Press 'q' to exit.")

while True:
    success, img = cap.read()
    if not success:
        print("Camera not found. Try changing 'camId' to 1 in the code.")
        break

    # 1. Detect Hand
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            # If we found a hand, get coordinates of Thumb (4) and Index (8)
            if len(lmList) != 0:
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb Tip
                x2, y2 = lmList[8][1], lmList[8][2]  # Index Tip
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Center point

                # 2. Draw visuals on the hand
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED) 
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED) 
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)      
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED) 

                # 3. Calculate Distance
                length = math.hypot(x2 - x1, y2 - y1)

                # 4. Convert Distance to Volume
                # Hand range 50-200 mapped to Volume range -65 to 0
                vol = np.interp(length, [50, 200], [minVol, maxVol])
                volBar = np.interp(length, [50, 200], [400, 150])
                volPer = np.interp(length, [50, 200], [0, 100])

                # 5. Set System Volume
                try:
                    volume.SetMasterVolumeLevel(vol, None)
                except:
                    pass # Ignore occasional volume errors

                # Visual Feedback: Green color if fingers are pinched
                if length < 50:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    # 6. Draw Volume Bar on Screen
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Volume Control", img)
    
    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
