import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def show_raw(path, w=512, h=512):
    img = np.fromfile(path, dtype=np.uint8).reshape((h, w))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(f"Raw Image: {os.path.basename(path)}")
    plt.show()

if __name__ == "__main__":
    try:
        image_width = int(sys.argv[2]) if len(sys.argv) > 2 else 512
        image_height = int(sys.argv[3]) if len(sys.argv) > 3 else image_width
    except:
        print("Invalid dimensions provided. Usage: python script.py [width] [height]")
        sys.exit(1)

    image_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')), sys.argv[1])
    show_raw(image_path, int(image_width), int(image_height))
