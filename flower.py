import cv2
import numpy as np
import math
class Flower:
    def __init__(self):
        self.petals = 8
    def draw_petal(self, img, center, angle, bloom, color):
        cx, cy = center
        length = int(40 + bloom * 65)
        width = int(12 + bloom * 12)
        offset = int(bloom * 30)
        bx = cx + int(math.cos(angle) * offset)
        by = cy + int(math.sin(angle) * offset)
        pts = []
        for t in np.linspace(0, math.pi, 25):
            x = width * math.sin(t)
            y = -length * math.cos(t)
            xr = x * math.cos(angle) - y * math.sin(angle)
            yr = x * math.sin(angle) + y * math.cos(angle)
            pts.append((int(bx + xr), int(by + yr)))
        for t in np.linspace(math.pi, 0, 25):
            x = -width * math.sin(t)
            y = -length * math.cos(t)
            xr = x * math.cos(angle) - y * math.sin(angle)
            yr = x * math.sin(angle) + y * math.cos(angle)
            pts.append((int(bx + xr), int(by + yr)))
        pts = np.array(pts, np.int32)
        # darker shade
        dark = (
            max(color[0]-40,0),
            max(color[1]-40,0),
            max(color[2]-40,0)
        )
        cv2.fillPoly(img,[pts],color)
        cv2.polylines(img,[pts],True,dark,2)
        cv2.polylines(img,[pts],True,(255,255,255),1)
    def draw_flower(self, frame, center, bloom, color):
        cx, cy = center
        for i in range(self.petals):
            angle = math.radians(i*360/self.petals)
            self.draw_petal(
                frame,
                center,
                angle,
                bloom,
                color
            )
        for i in range(self.petals):
            angle = math.radians(i*360/self.petals+15)
            self.draw_petal(
                frame,
                center,
                angle,
                bloom*0.8,
                color
            )
        cv2.circle(frame,center,18,(0,210,255),-1)
        cv2.circle(frame,center,10,(0,255,255),-1)
        for i in range(10):
            a = math.radians(i*36)
            x = int(cx+math.cos(a)*24)
            y = int(cy+math.sin(a)*24)
            cv2.line(frame,center,(x,y),(70,170,40),2)
            cv2.circle(frame,(x,y),3,(0,220,255),-1)
    def draw(self, frame, center, bloom):
        cx, cy = center
        colors = [
        (60,60,255),      # Red
        (0,255,255),      # Yellow
        (0,165,255),      # Orange
        (180,105,255),    # Pink
        (255,0,180)       # Purple
    ]
        positions = [
        (cx-150, cy+30),   # Red
        (cx-70,  cy-70),   # Yellow
        (cx,     cy-120),  # Orange
        (cx+80,  cy-70),   # Pink
        (cx+160, cy+30)    # Purple
    ]

        pot_center = (cx, cy+230)
        for x, y in positions:
            cv2.line(
            frame,
            (pot_center[0], pot_center[1]-5),
            (x, y),
            (40,150,40),
            7
        )
        leaves = [
        (cx-105, cy+120, -35),
        (cx-45, cy+45, -25),
        (cx+55, cy+45, 25),
        (cx+120, cy+120, 35),
        (cx-60, cy+175, -20),
        (cx+70, cy+175, 20)
    ]
        for lx, ly, angle in leaves:
            cv2.ellipse(
            frame,
            (lx,ly),
            (18,45),
            angle,
            0,
            360,
            (50,180,50),
            -1
        )
        for pos, color in zip(positions, colors):
          self.draw_flower(
            frame,
            pos,
            bloom,
            color
        )
        pot = np.array([
        [cx-35, cy+190],
        [cx+35, cy+190],
        [cx+25, cy+240],
        [cx-25, cy+240]
    ], np.int32)
        cv2.fillPoly(frame,[pot],(180,0,220))
        cv2.line(
        frame,
        (cx-28,cy+197),
        (cx+28,cy+197),
        (255,80,255),
        2
    )