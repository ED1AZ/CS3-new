import roboflow
import dotenv
import os
from sklearn.model_selection import KFold
import shutil

# directory from roboflow dataset
img_dir = "litter/images"
lbl_dir = "litter/labels"

# k = 10 --> 90-10 split
images = sorted(os.listdir(img_dir))
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=0)

dotenv.load_dotenv("keys.env")
KEY = os.getenv("ED1AZ_API_KEY")

fold_index = 1
for train_index, val_index in kf.split(images):
    # make kfold directories
    fold_path = f"kfold_{fold_index}"
    os.makedirs(f"{fold_path}/images/train", exist_ok=True)
    os.makedirs(f"{fold_path}/images/val", exist_ok=True)
    os.makedirs(f"{fold_path}/labels/train", exist_ok=True)
    os.makedirs(f"{fold_path}/labels/val", exist_ok=True)

    # copy images + labels
    for i in train_index:
        img = images[i]
        shutil.copy(f"{img_dir}/{img}", f"{fold_path}/images/train/{img}")
        shutil.copy(f"{lbl_dir}/{img.replace('.jpg','.txt')}",
                    f"{fold_path}/labels/train/{img.replace('.jpg','.txt')}")
    
    for i in val_index:
        img = images[i]
        shutil.copy(f"{img_dir}/{img}", f"{fold_path}/images/val/{img}")
        shutil.copy(f"{lbl_dir}/{img.replace('.jpg','.txt')}",
                    f"{fold_path}/labels/val/{img.replace('.jpg','.txt')}")
    
    # write data.yaml for binary classification
    with open(f"{fold_path}/data.yaml", "w") as f:
        f.write(
        f"train: {fold_path}/train/images\n"
        f"val: {fold_path}/valid/images\n"
        "nc: 1\n"
        "names: ['Trash']\n"
        )
    
    # CLASSES FOR BELOW MAY CHANGE AFTER MEETING

    # write data.yaml for material characterization
    # with open(f"{fold_path}/data.yaml", "w") as f:
    #     f.write(
    #     f"train: {fold_path}/train/images\n"
    #     f"val: {fold_path}/valid/images\n"
    #     "nc: 1\n"
    #     "names: ['Trash']\n"
    #     )
            
    # # write data.yaml for shape characterization
    # with open(f"{fold_path}/data.yaml", "w") as f:
    #     f.write(
    #     f"train: {fold_path}/train/images\n"
    #     f"val: {fold_path}/valid/images\n"
    #     "nc: 1\n"
    #     "names: ['Trash']\n"
    #     )
    
    fold_index += 1