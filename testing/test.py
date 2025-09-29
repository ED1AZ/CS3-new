import numpy as np
import cv2 as cv

def absdiff_with_threshold(src1, src2, threshold=30):
    src1 = np.asarray(src1, dtype=np.int16)
    src2 = np.asarray(src2, dtype=np.int16)

    diff = np.abs(src1 - src2)

    # Apply threshold: keep only pixels where difference > threshold
    mask = diff > threshold

    # Option 1: Keep original difference values where mask is true, else 0
    result = np.zeros_like(diff, dtype=np.uint8)
    result[mask] = diff[mask].astype(np.uint8)

    return result