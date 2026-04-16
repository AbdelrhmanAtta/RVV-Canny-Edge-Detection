import os
import sys
import numpy as np
import matplotlib.pyplot as plt

class TestImageGenerator:
    def __init__(self, size=256):
        self.size = size
        self.background = np.zeros((self.size, self.size), dtype=np.uint8)

    def rectangle(self, top_left=(50, 50), bottom_right=(200, 200)):
        img = self.background.copy()
        img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
        return img
    
    def circle(self, top_left=(50, 50), bottom_right=(200, 200)):
        img = self.background.copy()
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2
        radius = (bottom_right[0] - top_left[0]) // 2
        Y, X = np.ogrid[:self.size, :self.size]
        mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
        img[mask] = 255
        return img
    
    def diagonal_edge(self, top_left=(50, 50), bottom_right=(200, 200)):
        img = self.background.copy()
        y_start, y_end = top_left[1], bottom_right[1]
        x_start, x_end = top_left[0], bottom_right[0]
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                if (x - x_start) > (y - y_start):
                    img[y, x] = 255
        return img


if __name__ == "__main__":
    try:
        image_width = int(sys.argv[1]) if len(sys.argv) > 1 else 512
        image_height = int(sys.argv[2]) if len(sys.argv) > 2 else image_width
    except:
        print("Invalid dimensions provided. Usage: python script.py [width] [height]")
        sys.exit(1)

    gen = TestImageGenerator(size=max(image_height,image_width))

    assets_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets'))
    if not os.path.exists(assets_path):
        os.makedirs(assets_path)

    images = {
        "rect": gen.rectangle()[:image_height, :image_width],
        "circ": gen.circle()[:image_height, :image_width], 
        "diag": gen.diagonal_edge()[:image_height, :image_width]
    }

    for filename, data in images.items():
        filepath = os.path.join(assets_path, f"{filename}.raw")
        data.tofile(filepath)
