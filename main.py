import cv2 as cv

from src.blur import Blur
from src.face_detector import FaceDetector


def main():
    vid = cv.VideoCapture(0)
    face_detector = FaceDetector()

    while True:
        ret, frame = vid.read()

        faces = face_detector.process_frame(frame)

        for face in faces:
            x_min, y_min, x_max, y_max = face

            frame[y_min:y_max, x_min:x_max] = Blur.pixel_blur(
                image=frame[y_min:y_max, x_min:x_max],
                n_blocks=15
            )
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
