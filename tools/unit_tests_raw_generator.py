import os
import sys
import numpy as np
import matplotlib.pyplot as plt

## @class UnitTestsRawGenerator
#  @brief A utility class for generating basic geometric unit test raw images as NumPy arrays.
#
#  This class provides methods to create binary images (0 or 255) containing
#  rectangles, circles, and diagonal edges for testing image processing algorithms.
class UnitTestsRawGenerator:
    
    ## @brief Initializes the generator with a square canvas size.
    #  @param size The dimension (width and height) of the square background.
    def __init__(self, size=256):
        self.size = size
        self.background = np.zeros((self.size, self.size), dtype=np.uint8)

    ## @brief Generates a filled white rectangle on a black background.
    #  @param top_left A tuple (x, y) for the upper-left coordinate.
    #  @param bottom_right A tuple (x, y) for the lower-right coordinate.
    #  @return A NumPy array representing the generated image.
    def rectangle(self, top_left=(50, 50), bottom_right=(200, 200)):
        img = self.background.copy()
        img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
        return img
    
    ## @brief Generates a filled white circle on a black background.
    #  @param top_left The upper-left bound of the circle's bounding box.
    #  @param bottom_right The lower-right bound of the circle's bounding box.
    #  @return A NumPy array containing the circle.
    #  @note The radius is calculated based on the horizontal width provided.
    def circle(self, top_left=(50, 50), bottom_right=(200, 200)):
        img = self.background.copy()
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2
        radius = (bottom_right[0] - top_left[0]) // 2
        Y, X = np.ogrid[:self.size, :self.size]
        mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
        img[mask] = 255
        return img
    
    ## @brief Generates a diagonal edge (triangle) within a specified bounding box.
    #  @param top_left The upper-left corner of the region of interest.
    #  @param bottom_right The lower-right corner of the region of interest.
    #  @return A NumPy array with a 45-degree diagonal edge.
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
    ## @details Entry point for the script. 
    #  Parses command line arguments for image dimensions and saves
    #  generated assets to a '../assets' directory in raw binary format.
    try:
        image_width = int(sys.argv[1]) if len(sys.argv) > 1 else 512
        image_height = int(sys.argv[2]) if len(sys.argv) > 2 else image_width
    except:
        print("Invalid dimensions provided. Usage: python script.py [width] [height]")
        sys.exit(1)

    gen = UnitTestsRawGenerator(size=max(image_height,image_width))

    # Determine paths
    assets_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets'))
    if not os.path.exists(assets_path):
        os.makedirs(assets_path)

    # Generate image dictionary
    images = {
        "rect": gen.rectangle()[:image_height, :image_width],
        "circ": gen.circle()[:image_height, :image_width], 
        "diag": gen.diagonal_edge()[:image_height, :image_width]
    }

    # Save images as raw binary files
    for filename, data in images.items():
        filepath = os.path.join(assets_path, f"{filename}.raw")
        data.tofile(filepath)