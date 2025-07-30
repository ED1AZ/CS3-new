import cv2 as cv
import hg as hg

map = cv.imread('homography/maps/map.jpg')
cam = cv.VideoCapture("homography/videos/IMG_1086.mov")

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"Left button clicked at pixel coordinates: (x={x}, y={y})")
        # Get the homography matrix
        H = hg.get_matrix()
        
        # Transform the coordinates
        x_topdown, y_topdown = hg.transform_coordinates(x, y, H)
        # print(f"Transformed to top-down coordinates: (x={x_topdown}, y={y_topdown})")
        point_color = (0, 0, 255)
        # map[int(x_topdown), int(y_topdown)] = point_color
        cv.circle(map, (int(x_topdown), int(y_topdown)), 2, point_color, -1)



while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    cv.imshow("Camera Feed", frame)
    cv.imshow("Map", map)

    # Create a window and set the mouse callback function
    # cv.namedWindow('Image')
    cv.setMouseCallback('Camera Feed', mouse_callback)

    key = cv.waitKey(40)
    if key == ord('q'): # Press 'q' to quit
        break

cam.release()
cv.destroyAllWindows()
cv.imwrite("homography/maps/marked_map.jpg", map)
