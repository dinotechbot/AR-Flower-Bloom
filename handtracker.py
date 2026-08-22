import background
import cv2
import mediapipe as mp
import math
from flower import Flower
from spiderman import SpiderManMask
from background import BackgroundEffect
button_x1 = 30
button_y1 = 30
button_x2 = 230
button_y2 = 90
background_on = False
def mouse_callback(event, x, y, flags, param):
    global background_on
    if event == cv2.EVENT_LBUTTONDOWN:
        if (button_x1 <= x <= button_x2 and
                button_y1 <= y <= button_y2):
            background_on = not background_on
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)#open cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Simple Hand Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Simple Hand Tracker", 1600, 900)
cv2.setMouseCallback(
    "Simple Hand Tracker",
    mouse_callback
)
flower = Flower()#creating flower object
spiderman = SpiderManMask("spiderman_mask.png")
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frame = spiderman.draw(frame)
    if background_on:
     gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )
     frame = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )
    if background_on:
      button_color = (0, 180, 0)
      text = "B&W ON"
    else:
      button_color = (80, 80, 80)
      text = "B&W OFF"

    cv2.rectangle(
    frame,
    (button_x1, button_y1),
    (button_x2, button_y2),
    button_color,
    -1
)
    cv2.rectangle(
    frame,
    (button_x1, button_y1),
    (button_x2, button_y2),
    (255, 255, 255),
    2
)
    cv2.putText(
    frame,
    text,
    (button_x1 + 35, button_y1 + 38),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
      for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
        ):
        hand_label = handedness.classification[0].label
        mp_draw.draw_landmarks( #draw hand
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
        )
        landmarks = []
        h, w, _ = frame.shape
        for lm in hand_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))
        wrist = landmarks[0]
        thumb = landmarks[4]
        index = landmarks[8]
        tips = [4, 8, 12, 16, 20]
        total = 0
        for tip in tips:
            total += distance(wrist, landmarks[tip])
        average = total/5
        MIN_OPEN = 70
        MAX_OPEN = 220
        bloom = (average - MIN_OPEN) / (MAX_OPEN - MIN_OPEN)
        bloom = max(0, min(1, bloom))
        percent = int(bloom * 100)
        if hand_label == "Left":
            length = distance(thumb, index)
            MIN_DIST = 20
            MAX_DIST = 180
            brightness = (length - MIN_DIST)/(MAX_DIST - MIN_DIST)
            brightness = max(0.3, min(2.0, brightness * 2))
            frame = cv2.convertScaleAbs( #apply brightness
             frame,
             alpha=brightness,
             beta=0
            )
            cv2.line(frame, thumb, index, (255,0,255), 3)
            cv2.circle(frame, thumb, 8, (0,255,0), -1)
            cv2.circle(frame, index, 8, (0,255,0), -1)
        if hand_label=="Left":#draw line
         cv2.line(frame, thumb, index, (255,0,255), 3)
         cv2.circle(frame, thumb, 8, (0,255,0), -1)
         cv2.circle(frame, index, 8, (0,255,0), -1)
         if results.multi_hand_landmarks:
          h, w, _ = frame.shape
        if hand_label=="Right":
           flower.draw(
            frame,
            (w - 180, h // 2),
            bloom
           )
    cv2.imshow("Simple Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
           break
cap.release()
spiderman.close()
cv2.destroyAllWindows()