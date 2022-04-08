import cv2 as cv
import numpy as np


class Blur:

    @staticmethod
    def pixel_blur(image: np.array, n_blocks=10):

        image_height, image_width = image.shape[:2]
        x_s = np.linspace(
            start=0,
            stop=image_width,
            num=n_blocks + 1,
            dtype="int"
        )
        y_s = np.linspace(
            start=0,
            stop=image_height,
            num=n_blocks + 1,
            dtype="int"
        )

        for i in range(1, len(y_s)):
            for j in range(1, len(x_s)):
                x_min, x_max = x_s[j - 1], x_s[j]
                y_min, y_max = y_s[i - 1], y_s[i]

                roi = image[y_min:y_max, x_min:x_max]
                (B, G, R) = [int(x) for x in cv.mean(roi)[:3]]
                cv.rectangle(
                    img=image,
                    pt1=(x_min, y_min),
                    pt2=(x_max, y_max),
                    color=(B, G, R),
                    thickness=-1
                )
        return image
