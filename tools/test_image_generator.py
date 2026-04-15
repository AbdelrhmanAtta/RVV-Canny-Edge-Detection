import os
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
        return img
    
    def diagonal_edge(self, top_left=(50, 50), bottom_right=(200, 200)):
        img = self.background.copy()
        return img

H, W = 512, 512
gen = TestImageGenerator(size=H)

script_path = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.abspath(os.path.join(script_path, '../assets'))

if not os.path.exists(assets_path):
    os.makedirs(assets_path)

images = {
    "rect": gen.rectangle(), 
    "circ": gen.circle(), 
    "diag": gen.diagonal_edge(), 
}

loaded_data = {}
for filename, data in images.items():
    filepath = os.path.join(assets_path, f"{filename}.raw")
    data.tofile(filepath)
    loaded_data[filename] = np.fromfile(os.path.join(assets_path, 'rect.raw'), dtype=np.uint8).reshape(H, W)

plt.imshow(loaded_data["rect"], cmap='gray')
plt.show()
