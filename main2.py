# basic libraries
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv

# interface imports
from homography.int_funcs import init, transform_coordinates, get_matrix, \
    get_cpoints, get_vpoints, new_camera, open_camera, get_map, get_cameraview, update_img, \
    rotate_mapview, get_image, mouse_callback, inference

from bgsub.bgsub import VideoObject, findObjects, bgsubtract, saveROI, \
    findBinaryDiff, eliminateDuplicates, isolateGround, areasOfInterest, defineObjects,\
          getBackgroundModel

from bgsub.preprocessing import preprocessing

root = tk.Tk()
root.title("Homography Interface")
root.geometry("900x700")

root.rowconfigure(0, minsize=50, pad=10)
root.columnconfigure(1, minsize = 100)

init()

upload_button = tk.Button(root, text="Upload Map", command=get_map)
upload_button.grid(row=0, column=0, sticky = 'nesw')

upload_camview = tk.Button(root, text="Upload Camera View", command=get_image)
upload_camview.grid(row=0, column=1, sticky = 'news')

camera_button = tk.Button(root, text="Upload Camera", command=new_camera)
camera_button.grid(row=0, column=2, sticky = 'news')

open_button = tk.Button(root, text="Open Camera", command=open_camera)  # Open default camera
open_button.grid(row=0, column=3, sticky = 'news')

matv_button = tk.Button(root, text="Select Video Points", command=get_vpoints)
matv_button.grid(row=0, column=4, sticky = 'news')

matc_button = tk.Button(root, text="Select Image Points", command=get_cpoints)
matc_button.grid(row=0, column=5, sticky = 'news')

inf_button = tk.Button(root, text = "Inference", command = inference)
inf_button.grid(row=0, column = 6, sticky = 'news')

image_path = "homography/images/noimage.jpg"
original_image = Image.open(image_path)
original_image = original_image.resize((700, 500))  # Resize for display
tk_image = ImageTk.PhotoImage(original_image)

image_label = tk.Label(root, image = tk_image)
image_label.grid(row=1, column=0, columnspan=7)

root.mainloop()

print("Hello")