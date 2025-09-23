from pillow_heif import register_heif_opener
from PIL import Image
import os

# Register HEIF/HEIC support with Pillow
register_heif_opener()

# Folder with HEIC images
heic_folder = "parklitter/"
jpg_folder = "parklitter_jpg/"

# Make sure output folder exists
os.makedirs(jpg_folder, exist_ok=True)

# Loop through all HEIC files
for filename in os.listdir(heic_folder):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(heic_folder, filename)
        jpg_path = os.path.join(jpg_folder, filename.rsplit(".", 1)[0] + ".jpg")
        
        # Open HEIC file directly with Pillow after registering plugin
        image = Image.open(heic_path)
        
        # Save as JPG
        image.save(jpg_path, "JPEG")
        print(f"Converted {filename} -> {jpg_path}")