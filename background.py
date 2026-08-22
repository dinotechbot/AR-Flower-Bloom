import cv2


class BackgroundEffect:

    def __init__(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled

    def apply(self, frame):

        if not self.enabled:
            return frame

        # Convert background to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Convert back to BGR
        # so the rest of the program works normally
        frame = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        return frame