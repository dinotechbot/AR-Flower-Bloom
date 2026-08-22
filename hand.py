import cv2
import mediapipe as mp
import math


class HandTracker:

    def __init__(self):

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.drawer = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHand(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.result = self.hands.process(rgb)

        self.landmarks = []

        if self.result.multi_hand_landmarks:

            hand = self.result.multi_hand_landmarks[0]

            self.drawer.draw_landmarks(
                frame,
                hand,
                self.mpHands.HAND_CONNECTIONS,
                self.drawer.DrawingSpec(
                    color=(0,255,0),
                    thickness=2,
                    circle_radius=2
                ),
                self.drawer.DrawingSpec(
                    color=(255,255,255),
                    thickness=2
                )
            )

            h,w,_ = frame.shape

            for lm in hand.landmark:

                x = int(lm.x*w)
                y = int(lm.y*h)

                self.landmarks.append((x,y))

        return frame

    def indexTip(self):

        if len(self.landmarks)==0:
            return None

        return self.landmarks[8]

    def fingersUp(self):

        if len(self.landmarks)==0:
            return []

        fingers=[]

        # Thumb
        if self.landmarks[4][0] > self.landmarks[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for tip in [8,12,16,20]:

            if self.landmarks[tip][1] < self.landmarks[tip-2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def indexOnly(self):

        f=self.fingersUp()

        if len(f)!=5:
            return False

        return f==[0,1,0,0,0]

    def palmOpen(self):

        f=self.fingersUp()

        if len(f)!=5:
            return False

        return sum(f)>=4

    def openness(self):

        if len(self.landmarks)==0:
            return 0

        wrist=self.landmarks[0]

        total=0

        for i in [4,8,12,16,20]:

            tip=self.landmarks[i]

            total+=math.hypot(
                tip[0]-wrist[0],
                tip[1]-wrist[1]
            )

        avg=total/5

        bloom=(avg-70)/(210-70)

        bloom=max(0,min(1,bloom))

        return bloom