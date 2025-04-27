
import cv2


def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    image = image.copy()
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), 3, color, -1)
    return image