import cv2 as cv
import numpy as np
from openvino.inference_engine import IECore


class FaceDetector:
    def __init__(self, conf_threshold=0.9):
        model = "models/face_detector/face-detection-0204.xml"
        config = "models/face_detector/face-detection-0204.bin"

        self.ie = IECore()
        self.network = self.ie.read_network(model=model, weights=config)
        self.executable_network = self.ie.load_network(self.network, device_name='CPU', num_requests=1)
        self.input_name = next(iter(self.network.input_info))

        self.conf_threshold = conf_threshold

    def process_frame(self, frame: np.array):
        frame_height, frame_width = frame.shape[:2]

        resized = cv.resize(frame, (448, 448), interpolation=cv.INTER_AREA)
        resized = resized.transpose(2, 0, 1)
        nn_outputs = self.executable_network.infer({self.input_name: resized})
        faces_raw = nn_outputs["detection_out"][0][0]

        faces = []
        for (_, label, conf, x_min, y_min, x_max, y_max) in faces_raw:
            if self.conf_threshold <= conf:
                x_min = abs(int(x_min * frame_width))
                x_max = abs(int(x_max * frame_width))
                y_min = abs(int(y_min * frame_height))
                y_max = abs(int(y_max * frame_height))

                faces.append([x_min, y_min, x_max, y_max])

        return faces
