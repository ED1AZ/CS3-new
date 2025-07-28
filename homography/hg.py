import cv2 as cv
import numpy as np

def get_matrix():
    # Hardcoded points to form homography matrix with
    video_points = np.array([[243, 243], [761, 231], [234, 414], [560, 430]]) # from camera perspective
    topdown_points = np.array([[68, 259], [76, 147], [168, 276], [175, 268]]) # from top-down perspective
    # Computing homography matrix
    H, _ = cv.findHomography(video_points, topdown_points, method=cv.RANSAC)
    return H

# To transform coordinates to desired perspective, input is x and y coordinate of the point to be transformed + homography matrix
def transform_coordinates(x_vid, y_vid, H):
    point = np.array([[x_vid, y_vid]], dtype=np.float32).reshape(1, 1, 2)
    transformed_point = cv.perspectiveTransform(point, H)
    return transformed_point[0][0]
