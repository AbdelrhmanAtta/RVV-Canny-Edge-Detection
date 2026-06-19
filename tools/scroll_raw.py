import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re


class PipelineViewer:
    def __init__(self, assets_dir, filter_str):
        self.files = self._auto_gather(assets_dir, filter_str)
        self.current_idx = 0
        self.playing = False
        if not self.files:
            print(f"Error: No files found matching '{filter_str}'")
            sys.exit(1)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        print(f"Loaded {len(self.files)} pipeline stages.")
        self.update_display()
        plt.show()

    def _auto_gather(self, assets_dir, filter_str):
        # The specific order of the Canny pipeline
        stage_priority = {
            "blur": 1, "spatial": 1, "sep": 2, "gx": 3, "gy": 4,
            "mag": 5, "mag1": 5, "mag2": 5, "mag_l1": 5, "mag_l2": 5,
            "dir": 6, "nms": 7, "thresh": 8, "double_thresholding": 8,
            "hysteresis": 9, "canny": 10
        }
        stages_by_len = sorted(stage_priority, key=len, reverse=True)

        all_files = glob.glob(os.path.join(assets_dir, f"*{filter_str}*.raw"))

        def get_sort_key(filepath):
            filename = os.path.basename(filepath)
            base = filename[:-4] if filename.endswith(".raw") else filename
            # suffix style: rvv_circ312x444_gx.raw
            for stage in stages_by_len:
                if base.endswith("_" + stage):
                    return stage_priority[stage]
            # prefix style: gx_Drone_638x480_gray.raw
            for stage in stages_by_len:
                if base.startswith(stage + "_"):
                    return stage_priority[stage]
            return 0  # base/original file with no stage tag

        all_files.sort(key=get_sort_key)
        return [(os.path.basename(f), f) for f in all_files]

    def load_raw(self, path):
        filename = os.path.basename(path)
        # Parse dimensions
        dims = re.search(r'(\d+)x(\d+)', filename)
        w, h = (int(dims.group(1)), int(dims.group(2))) if dims else (1, 1)

        # Determine bit-depth and signedness
        is_16bit = (os.path.getsize(path) == w * h * 2)
        is_signed = any(x in filename for x in ['gx', 'gy'])
        dtype = np.int16 if (is_16bit and is_signed) else (np.uint16 if is_16bit else (np.int8 if is_signed else np.uint8))

        img = np.fromfile(path, dtype=dtype).reshape((h, w))

        # Normalization
        if 'gx' in filename or 'gy' in filename:
            vals, counts = np.unique(img, return_counts=True)
            bg = float(vals[np.argmax(counts)])  # background/mode value
            diff = img.astype(np.float32) - bg
            absmax = np.abs(diff).max() + 1e-5
            img = diff / absmax * 127 + 128
        elif img.max() != img.min():
            img = ((img.astype(np.float32) - img.min()) / (img.max() - img.min()) * 255)

        return img.astype(np.uint8), f"{w}x{h}"

    def update_display(self):
        self.ax.clear()
        label, fpath = self.files[self.current_idx]
        img, dims = self.load_raw(fpath)
        self.ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        self.ax.set_title(f"[{self.current_idx + 1}/{len(self.files)}] {label}\n{dims}")
        self.ax.axis('off')
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % len(self.files)
            self.update_display()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.files)
            self.update_display()
        elif event.key == ' ':
            self.playing = not self.playing
            while self.playing:
                self.current_idx = (self.current_idx + 1) % len(self.files)
                self.update_display()
                plt.pause(0.5)


if __name__ == "__main__":
    # Usage: python scroll_raw.py <folder> <filter>
    # Example: python scroll_raw.py tests rvv_circ
    mode = sys.argv[1] if len(sys.argv) > 1 else "assets"
    filter_str = sys.argv[2] if len(sys.argv) > 2 else ""
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests_assets' if mode == 'tests' else '../assets'))
    viewer = PipelineViewer(assets_dir, filter_str)