import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def show_raw(path, w=512, h=512):
    img = np.fromfile(path, dtype=np.uint8).reshape((h, w))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(f"Raw Image: {os.path.basename(path)}")
    plt.show()

if name == "main":
    image_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(file)), '../assets')), sys.argv[1])
    show_raw(image_path, int(sys.argv[2]), int(sys.argv[3]))
