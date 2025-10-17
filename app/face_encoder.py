import numpy as np
import cv2
import insightface

class FaceEncoder:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def l2_normalize(self, x):
        return x / np.sqrt(np.sum(np.square(x)))

    def encode_image(self, file_bytes):
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            print("⚠️ Could not read image bytes")
            return None

        faces = self.app.get(img)
        if len(faces) == 0:
            print("⚠️ No face detected in uploaded image")
            return None

        return self.l2_normalize(faces[0].embedding)
