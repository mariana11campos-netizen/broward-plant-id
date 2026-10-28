"""USB webcam capture, plus a fake camera for testing without hardware."""
import numpy as np


class Webcam:
    def __init__(self, cfg):
        import cv2

        self.cv2 = cv2
        c = cfg["camera"]
        self.cap = cv2.VideoCapture(c["device_index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        self.warmup = c["warmup_frames"]
        if not self.cap.isOpened():
            raise RuntimeError("Webcam not found. Check the USB cable and camera.device_index.")

    def capture(self):
        """Returns an RGB numpy image. Discards a few frames so auto-exposure and focus settle."""
        for _ in range(self.warmup):
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Webcam returned no frame.")
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class MockWebcam:
    """Random green leaf-ish blobs on a brown background. Only useful for testing the pipeline."""

    def __init__(self, cfg, seed=None):
        self.rng = np.random.default_rng(seed)

    def capture(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (110, 85, 60)
        for _ in range(self.rng.integers(3, 8)):
            cy, cx = self.rng.integers(60, 420), self.rng.integers(60, 580)
            ry, rx = self.rng.integers(20, 80), self.rng.integers(30, 120)
            yy, xx = np.ogrid[:480, :640]
            mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            img[mask] = self.rng.integers((30, 90, 20), (90, 180, 70))
        return img

    def close(self):
        pass
