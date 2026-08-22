import cv2
from hand import HandTracker
from stem import Stem
from flower import Flower

tracker = HandTracker()
stem = Stem()
flower = Flower()

cap = cv2.VideoCapture(0)

# States
DRAW = 0
BLOOM = 1

state = DRAW

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    frame = tracker.findHand(frame)

    # -------------------------
    # DRAW STEM
    # -------------------------

    if state == DRAW:

        if tracker.indexOnly():

            tip = tracker.indexTip()

            if tip is not None:

                stem.add_point(tip)

        stem.draw(frame)

        if len(stem.points) > 50:

            stem.finished = True

            state = BLOOM

    # -------------------------
    # BLOOM
    # -------------------------

    else:

        stem.draw(frame)

        bloom = tracker.openness()

        tip = stem.tip()

        if tip is not None:

            flower.draw(
                frame,
                tip,
                bloom
            )

    cv2.imshow("Magic Bloom",frame)

    key=cv2.waitKey(1)

    if key==ord('r'):

        stem.reset()

        state=DRAW

    if key==27:

        break

cap.release()

cv2.destroyAllWindows()