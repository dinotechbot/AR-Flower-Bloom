import cv2
import math


class Stem:

    def __init__(self):

        self.points = []

        self.finished = False

    def add_point(self, point):

        if self.finished:
            return

        if len(self.points) == 0:

            self.points.append(point)

        else:

            last = self.points[-1]

            d = math.hypot(point[0]-last[0],
                           point[1]-last[1])

            # avoid duplicate points
            if d > 5:

                self.points.append(point)

    def draw(self, frame):

        if len(self.points) < 2:
            return

        for i in range(len(self.points)-1):

            cv2.line(
                frame,
                self.points[i],
                self.points[i+1],
                (40,170,40),
                7
            )

        # leaves every few points
        for i in range(5, len(self.points), 18):

            x, y = self.points[i]

            cv2.ellipse(
                frame,
                (x-12,y),
                (10,25),
                -35,
                0,
                360,
                (60,180,60),
                -1
            )

            cv2.ellipse(
                frame,
                (x+12,y+5),
                (10,25),
                35,
                0,
                360,
                (60,180,60),
                -1
            )

    def tip(self):

        if len(self.points)==0:
            return None

        return self.points[-1]

    def reset(self):

        self.points=[]

        self.finished=False