import sys
import os
import numpy as np
import matplotlib.pyplot as plt

## @brief Reads and displays a raw binary image file using Matplotlib.
#
#  This function assumes the data is stored as unsigned 8-bit integers (uint8)
#  and requires the user to specify the correct dimensions for reshaping.
#
#  @param path The absolute or relative path to the .raw image file.
#  @param w    The width of the image in pixels. Defaults to 512.
#  @param h    The height of the image in pixels. Defaults to 512.
#
#  @return None. Displays a grayscale plot of the image.
def show_raw(path, w=512, h=512):
    try:
        # Load the binary data and reshape into a 2D grid
        img = np.fromfile(path, dtype=np.uint8).reshape((h, w))
        
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Raw Image: {os.path.basename(path)}")
        plt.axis('off')  # Hide axes for a cleaner look
        plt.show()
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
    except ValueError as e:
        print(f"Error: Could not reshape data. Ensure dimensions {w}x{h} are correct. {e}")

if __name__ == "__main__":
    ## @details Main execution block for the visualization script.
    #  Expects the filename as the first argument, with optional width and height.
    #  Usage: python script.py <filename.raw> [width] [height]
    
    if len(sys.argv) < 2:
        print("Usage: python script.py [filename] [width] [height]")
        sys.exit(1)

    try:
        # Parse dimensions from CLI arguments
        image_width = int(sys.argv[2]) if len(sys.argv) > 2 else 512
        image_height = int(sys.argv[3]) if len(sys.argv) > 3 else image_width
    except ValueError:
        print("Invalid dimensions provided. Width and Height must be integers.")
        sys.exit(1)

    # Construct the path to the assets folder relative to this script
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets'))
    image_path = os.path.join(assets_dir, sys.argv[1])
    
    show_raw(image_path, int(image_width), int(image_height))