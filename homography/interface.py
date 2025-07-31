import tkinter as tk
import cv2 as cv
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np

global H
H = np.zeros((3, 3), dtype=np.float64)

global clicks
clicks = 0

root = tk.Tk()
root.title("Homography Interface")
root.geometry("900x700")

cams = [None] * 10 # List to hold camera objects
markedmap = "homography/maps/markedmap.jpg"
image = "homography/images/image.jpg"

# video_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]]) # from camera perspective
# topdown_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]]) # from top-down perspective
video_points = np.array([[243, 243], [761, 231], [234, 414], [560, 430]]) # from camera perspective
topdown_points = np.array([[68, 259], [76, 147], [168, 276], [175, 268]])

def transform_coordinates(x_vid, y_vid):
    global H
    point = np.array([[x_vid, y_vid]], dtype=np.float32).reshape(1, 1, 2)
    transformed_point = cv.perspectiveTransform(point, H)
    return transformed_point[0][0]

def get_matrix():
    global video_points, topdown_points
    global H
    H, _ = cv.findHomography(video_points, topdown_points, method=cv.RANSAC)
    print("Homography matrix computed:")
    print(H)

def click_event(event, x, y, flags, param):
    global clicks

    arr = param
    if event == cv.EVENT_LBUTTONDOWN:
        arr[clicks] = [x, y]
        clicks += 1
        print(f"Clicked at (x: {x}, y: {y})")

def get_cpoints(): # topdown points, c points
    # Placeholder for homography matrix retrieval logic
    global clicks
    clicks = 0
    temp_img = Image.open(markedmap)
    temp_img = np.array(temp_img)
    temp_img = cv.cvtColor(temp_img, cv.COLOR_RGB2BGR)

    cv.imshow('Map', temp_img)
    cv.setMouseCallback('Map', click_event, topdown_points)

    while True:
        cv.imshow('Map', temp_img)
        key = cv.waitKey(1) & 0xFF

        if clicks >= 4:
            break

    cv.destroyAllWindows()
    clicks = 0

def get_vpoints(): # camera view, video
    # Placeholder for homography matrix retrieval logic
    global clicks
    clicks = 0
    temp_img = Image.open(image)
    temp_img = np.array(temp_img)
    temp_img = cv.cvtColor(temp_img, cv.COLOR_RGB2BGR)

    cv.imshow('Video', temp_img)
    cv.setMouseCallback('Video', click_event, video_points)

    while True:
        cv.imshow('Video', temp_img)
        key = cv.waitKey(1) & 0xFF

        if clicks >= 4:
            break
    cv.destroyAllWindows()
    clicks = 0


def new_camera():
    user_input = simpledialog.askstring("Input", "Enter Camera ID:")
    if user_input:
        print(f"Camera created: {user_input}")
    else:
        print("User cancelled or entered no text.")
        return

    open_button.pack(pady=20)
    cams[int(user_input)] = (cv.VideoCapture(user_input))

def open_camera():
    ind = simpledialog.askstring("Input", "Enter Camera ID:")
    if ind:
        print(f"Camera opened: {ind}")
    else:
        print("User cancelled or entered no text.")

    cam = cams[int(ind)]
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        cv.imshow("Camera Feed", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()

def get_map():
    global image_label, markedmap

    file_path = filedialog.askopenfilename(title="Select a map image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return None
    
    cv.imwrite(markedmap, cv.imread(file_path))

    tk_image = Image.open(markedmap)
    tk_image = tk_image.resize((200, 300))  # Resize for display

    tk_image = ImageTk.PhotoImage(tk_image)
    image_label.config(image=tk_image)

    image_label.image = tk_image  # Keep a reference to avoid garbage collection

    # upload_button.grid_forget()
    rotate_button.grid(row=6, column=0, padx=10, pady=20)
    messagebox.showinfo("Success", "Map uploaded successfully.")

def get_cameraview(ind):
    file_path = filedialog.askopenfilename(title="Select a camera image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return None
    
    cv.imwrite(markedmap, cv.imread(file_path))

def update_img():
    global image_label, markedmap

    tk_image = Image.open(markedmap)
    tk_image = tk_image.resize((200, 300))
    tk_image = ImageTk.PhotoImage(tk_image)
    image_label.config(image=tk_image)

    image_label.image = tk_image
    image_label.update_idletasks()

def rotate_mapview():
    global markedmap
    map_image = cv.imread(markedmap)
    rotated_map = cv.rotate(map_image, cv.ROTATE_90_CLOCKWISE)
    cv.imwrite(markedmap, rotated_map)
    update_img()
    messagebox.showinfo("Success", "Map rotated successfully.")
    # Image.open(markedmap).show()  # Display the rotated map

def get_image():
    file_path = filedialog.askopenfilename(title="Select a camera view image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return None
    
    cv.imwrite(image, cv.imread(file_path))
    messagebox.showinfo("Success", "Camera View uploaded successfully.")

def mouse_callback(event, x, y, flags, param):
    global H

    if event == cv.EVENT_LBUTTONDOWN:
        map_image = cv.imread(markedmap)

        # Transform the coordinates
        x_topdown, y_topdown = transform_coordinates(x, y)
        point_color = (0, 0, 255)
        cv.circle(map_image, (int(x_topdown), int(y_topdown)), 5, point_color, -1)
        # cv.imshow("MAP", map_image)

        cv.imwrite(markedmap, map_image)
        update_img()
        print(f"Left button clicked at pixel coordinates: (x={x}, y={y})")
        
def inference():
    global H, video_points, topdown_points
    print(H)
    print(video_points)
    print(topdown_points)
    map = cv.imread('homography/maps/map.jpg')
    cam = cv.VideoCapture("homography/videos/IMG_1086.mov")

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        cv.imshow("Camera Feed", frame)
        cv.setMouseCallback('Camera Feed', mouse_callback)
        # update_img()

        key = cv.waitKey(40)
        if key == ord('q'): # Press 'q' to quit
            break

    cam.release()
    cv.destroyAllWindows()

# buttons
upload_button = tk.Button(root, text="Upload Map", command=get_map)
upload_button.grid(row=0, column=0, padx=10, pady=10)

upload_camview = tk.Button(root, text="Upload Camera View", command=get_image)
upload_camview.grid(row=1, column=0, padx=10, pady=10)

camera_button = tk.Button(root, text="Upload Camera", command=new_camera)
camera_button.grid(row=2, column=0, padx=10, pady=10)

open_button = tk.Button(root, text="Open Camera", command=open_camera)  # Open default camera

mat_button = tk.Button(root, text="Select Video Points", command=get_vpoints)
mat_button.grid(row=3, column=0, padx=10, pady=10)

mat_button = tk.Button(root, text="Select Image Points", command=get_cpoints)
mat_button.grid(row=4, column=0, padx=10, pady=10)

inf_button = tk.Button(root, text = "Inference", command = inference)
inf_button.grid(row=7, column = 0, padx = 10, pady = 10)

get_matrix = tk.Button(root, text="Get Homography Matrix", command=get_matrix)
get_matrix.grid(row=5, column=0, padx=10, pady=10)

rotate_button = tk.Button(root, text="Rotate Map", command=rotate_mapview)

# map display
image_path = "homography/images/noimage.jpg"
original_image = Image.open(image_path)
original_image = original_image.resize((700, 500))  # Resize for display
tk_image = ImageTk.PhotoImage(original_image)

image_label = tk.Label(root, image = tk_image)
image_label.grid(row=0, column=1, rowspan=6, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()