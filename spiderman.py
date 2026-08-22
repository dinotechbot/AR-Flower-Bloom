import cv2
import mediapipe as mp
import numpy as np
import math


class SpiderManMask:

    def __init__(self, mask_path="spiderman_mask.png"):

        self.mask = cv2.imread(
            mask_path,
            cv2.IMREAD_UNCHANGED
        )

        if self.mask is None:
            raise FileNotFoundError(
                "spiderman_mask.png not found!"
            )

        if self.mask.shape[2] != 4:
            raise ValueError(
                "spiderman_mask.png must be a transparent PNG!"
            )

        # -----------------------------
        # MediaPipe Face Mesh
        # -----------------------------

        self.mp_face = mp.solutions.face_mesh

        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # -----------------------------
        # Previous values
        # -----------------------------

        self.prev_x = None
        self.prev_y = None
        self.prev_width = None
        self.prev_height = None
        self.prev_angle = None

    # ==================================================
    # Smooth movement
    # ==================================================

    def smooth(self, previous, current, factor):

        if previous is None:
            return current

        return previous + (
            current - previous
        ) * factor

    # ==================================================
    # Rotate PNG
    # ==================================================

    def rotate_image(self, image, angle):

        h, w = image.shape[:2]

        center = (
            w // 2,
            h // 2
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])

        new_width = int(
            h * sin + w * cos
        )

        new_height = int(
            h * cos + w * sin
        )

        matrix[0, 2] += (
            new_width / 2 - center[0]
        )

        matrix[1, 2] += (
            new_height / 2 - center[1]
        )

        return cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    # ==================================================
    # Overlay transparent image
    # ==================================================

    def overlay(self, frame, overlay, x, y):

        frame_h, frame_w = frame.shape[:2]

        overlay_h, overlay_w = overlay.shape[:2]

        # --------------------------------
        # Clip to frame
        # --------------------------------

        x1 = max(0, int(x))
        y1 = max(0, int(y))

        x2 = min(
            frame_w,
            int(x + overlay_w)
        )

        y2 = min(
            frame_h,
            int(y + overlay_h)
        )

        if x1 >= x2 or y1 >= y2:
            return frame

        # --------------------------------
        # Overlay coordinates
        # --------------------------------

        ox1 = x1 - int(x)
        oy1 = y1 - int(y)

        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        overlay_crop = overlay[
            oy1:oy2,
            ox1:ox2
        ]

        # --------------------------------
        # Alpha blending
        # --------------------------------

        overlay_rgb = overlay_crop[:, :, :3]

        alpha = (
            overlay_crop[:, :, 3]
            .astype(np.float32)
            / 255.0
        )

        # Slight transparency
        alpha *= 0.96

        alpha = alpha[:, :, None]

        frame_crop = frame[
            y1:y2,
            x1:x2
        ]

        result = (
            overlay_rgb * alpha
            +
            frame_crop * (1 - alpha)
        )

        frame[
            y1:y2,
            x1:x2
        ] = result.astype(np.uint8)

        return frame

    # ==================================================
    # Main tracking
    # ==================================================

    def draw(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame

        face = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        # ==================================================
        # IMPORTANT FACE LANDMARKS
        # ==================================================

        # Left eye outer corner
        left_eye = face.landmark[33]

        # Right eye outer corner
        right_eye = face.landmark[263]

        # Nose
        nose = face.landmark[1]

        # Forehead
        forehead = face.landmark[10]

        # Chin
        chin = face.landmark[152]

        # ==================================================
        # Convert to pixels
        # ==================================================

        lx = left_eye.x * w
        ly = left_eye.y * h

        rx = right_eye.x * w
        ry = right_eye.y * h

        nx = nose.x * w
        ny = nose.y * h

        fx = forehead.x * w
        fy = forehead.y * h

        cx = chin.x * w
        cy = chin.y * h

        # ==================================================
        # Eye center
        # ==================================================

        eye_center_x = (
            lx + rx
        ) / 2

        eye_center_y = (
            ly + ry
        ) / 2

        # ==================================================
        # Distance between eyes
        # ==================================================

        eye_distance = math.sqrt(
            (rx - lx) ** 2
            +
            (ry - ly) ** 2
        )

        if eye_distance < 20:
            return frame

        # ==================================================
        # Calculate face angle
        # ==================================================

        angle = math.degrees(
            math.atan2(
                ry - ly,
                rx - lx
            )
        )

        # ==================================================
        # Calculate mask size
        #
        # Eye distance is more stable than
        # the complete face bounding box.
        # ==================================================

        mask_width = int(eye_distance * 1.65)
        mask_height = int(mask_width * 1.25)

        # ==================================================
        # Position
        #
        # Slightly above eye center
        # so mask sits naturally on face.
        # ==================================================

        target_x = eye_center_x

        target_y = (
           eye_center_y
           + eye_distance * 0.20
)
        # ==================================================
        # Smooth movement
        # ==================================================

        # Position
        self.prev_x = self.smooth(
            self.prev_x,
            target_x,
            0.40
        )

        self.prev_y = self.smooth(
            self.prev_y,
            target_y,
            0.40
        )

        # Size
        self.prev_width = self.smooth(
            self.prev_width,
            mask_width,
            0.35
        )

        self.prev_height = self.smooth(
            self.prev_height,
            mask_height,
            0.35
        )

        # Rotation
        if self.prev_angle is None:
            self.prev_angle = angle
        else:
            self.prev_angle = self.smooth(
                self.prev_angle,
                angle,
                0.35
            )

        # ==================================================
        # Final dimensions
        # ==================================================

        final_width = max(
            50,
            int(self.prev_width)
        )

        final_height = max(
            70,
            int(self.prev_height)
        )

        # ==================================================
        # Resize mask
        # ==================================================

        resized = cv2.resize(
            self.mask,
            (
                final_width,
                final_height
            ),
            interpolation=cv2.INTER_AREA
        )

        # ==================================================
        # Rotate with head
        # ==================================================

        rotated = self.rotate_image(
            resized,
            -self.prev_angle
        )

        rh, rw = rotated.shape[:2]

        # ==================================================
        # Final position
        # ==================================================

        x = int(
            self.prev_x - rw / 2
        )

        y = int(
            self.prev_y - rh / 2
        )

        # ==================================================
        # Draw mask
        # ==================================================

        frame = self.overlay(
            frame,
            rotated,
            x,
            y
        )

        return frame

    # ==================================================
    # Close
    # ==================================================

    def close(self):

        self.face_mesh.close()