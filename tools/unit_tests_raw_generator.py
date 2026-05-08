import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import unittest

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

    ## @brief Generates a horizontal white line on a black background.
    #  @param y The vertical position of the line.
    #  @param x_range A tuple (start, end) for the horizontal range.
    #  @return A NumPy array representing the horizontal line.
    def horizontal_line(self, y=100, x_range=(50, 200)):
        img = self.background.copy()
        img[y, x_range[0]:x_range[1]] = 255
        return img

    ## @brief Generates a vertical white line on a black background.
    #  @param x The horizontal position of the line.
    #  @param y_range A tuple (start, end) for the vertical range.
    #  @return A NumPy array representing the vertical line.
    def vertical_line(self, x=100, y_range=(50, 200)):
        img = self.background.copy()
        img[y_range[0]:y_range[1], x] = 255
        return img

    ## @brief Generates an inverse diagonal edge (135 degrees) within a specified bounding box.
    #  @param top_left The upper-left corner of the region of interest.
    #  @param bottom_right The lower-right corner of the region of interest.
    #  @return A NumPy array with a 135-degree diagonal edge.
    def diagonal_edge_inv(self, top_left=(50, 50), bottom_right=(200, 200)):
        img = self.background.copy()
        y_start, y_end = top_left[1], bottom_right[1]
        x_start, x_end = top_left[0], bottom_right[0]
        width = x_end - x_start
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                if (x - x_start) + (y - y_start) < width:
                    img[y, x] = 255
        return img


## @class TestUnitTestsRawGenerator
#  @brief Unit tests for the UnitTestsRawGenerator class.
class TestUnitTestsRawGenerator(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.gen = UnitTestsRawGenerator(size=self.size)

    def test_rectangle(self):
        img = self.gen.rectangle(top_left=(10, 10), bottom_right=(20, 20))
        self.assertEqual(img.shape, (self.size, self.size))
        self.assertEqual(img[15, 15], 255)
        self.assertEqual(img[5, 5], 0)

    def test_circle(self):
        img = self.gen.circle(top_left=(10, 10), bottom_right=(30, 30))
        # Center is at (20, 20), radius is 10
        self.assertEqual(img[20, 20], 255)
        self.assertEqual(img[0, 0], 0)

    def test_horizontal_line(self):
        img = self.gen.horizontal_line(y=50, x_range=(10, 90))
        self.assertEqual(img[50, 50], 255)
        self.assertEqual(img[51, 50], 0)

    def test_vertical_line(self):
        img = self.gen.vertical_line(x=50, y_range=(10, 90))
        self.assertEqual(img[50, 50], 255)
        self.assertEqual(img[50, 51], 0)

    def test_diagonal_edge(self):
        img = self.gen.diagonal_edge(top_left=(10, 10), bottom_right=(90, 90))
        # 45 degree: x-x0 > y-y0 -> 255
        self.assertEqual(img[20, 30], 255)  # 30-10 > 20-10 (20 > 10)
        self.assertEqual(img[30, 20], 0)    # 20-10 > 30-10 (10 > 20)

    def test_diagonal_edge_inv(self):
        img = self.gen.diagonal_edge_inv(top_left=(10, 10), bottom_right=(90, 90))
        # 135 degree: (x-x0) + (y-y0) < width -> 255
        # width = 80
        self.assertEqual(img[20, 20], 255)  # (10) + (10) < 80
        self.assertEqual(img[80, 80], 0)    # (70) + (70) < 80


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        sys.argv.pop(1)
        unittest.main()
    else:
        # Existing main logic
        try:
            image_width = int(sys.argv[1]) if len(sys.argv) > 1 else 512
            image_height = int(sys.argv[2]) if len(sys.argv) > 2 else image_width
        except:
            print("Invalid dimensions provided. Usage: python script.py [width] [height]")
            sys.exit(1)

        gen = UnitTestsRawGenerator(size=max(image_height, image_width))

        # Determine paths
        assets_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets'))
        if not os.path.exists(assets_path):
            os.makedirs(assets_path)

        # Generate image dictionary
        images = {
            "rect": gen.rectangle()[:image_height, :image_width],
            "circ": gen.circle()[:image_height, :image_width],
            "diag": gen.diagonal_edge()[:image_height, :image_width],
            "diag_inv": gen.diagonal_edge_inv()[:image_height, :image_width],
            "horiz": gen.horizontal_line()[:image_height, :image_width],
            "vert": gen.vertical_line()[:image_height, :image_width]
        }

        # Save images as raw binary files
        for filename, data in images.items():
            filepath = os.path.join(assets_path, f"{filename}.raw")
            data.tofile(filepath)
            print(f"Generated {filepath}")