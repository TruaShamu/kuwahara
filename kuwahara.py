import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class KuwaharaFilter:
    def __init__(self, window_size=5):
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("Pick an odd sized window_size >= 3")
        self.window_size = window_size
        self.half = window_size // 2
        self.img = None

    def load_img(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise ValueError("cv2.imread failed to load image")
        self.img = self.img_bgr.copy()
    
    def show_img(self, img=None, title="Image"):
        if img is None:
            img = self.img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.axis('off')
        plt.title(title)
        plt.imshow(rgb)
        plt.show()
    
    def save_img(self, out_path):
        cv2.imwrite(out_path, self.img)

    def to_hsv(self):
        # cv2.imread returns BGR images, so convert from BGR -> HSV
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

    def kuwahara(self):
        if self.img is None:
            raise RuntimeError("No image loaded")

        # Work on HSV value channel
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        v = v_ch.astype(np.float64)

        height, width = v.shape

        # Build integral images for v and v*v to compute mean and variance quickly
        # Pad with an extra zero row/col at the top/left for simpler area sums
        S = np.zeros((height + 1, width + 1), dtype=np.float64)
        SQ = np.zeros((height + 1, width + 1), dtype=np.float64)

        S[1:, 1:] = v.cumsum(axis=0).cumsum(axis=1)
        SQ[1:, 1:] = (v * v).cumsum(axis=0).cumsum(axis=1)

        def rect_sum(ii, r1, c1, r2, c2):
            # ii is integral image with 1-based padding, r1,c1,r2,c2 are inclusive 0-based
            # convert to 1-based indices for II: add 1
            return ii[r2 + 1, c2 + 1] - ii[r1, c2 + 1] - ii[r2 + 1, c1] + ii[r1, c1]

        out_v = np.zeros_like(v)

        half = self.half

        for r in range(height):
            r1 = max(0, r - half)
            r2 = min(height - 1, r + half)
            for c in range(width):
                c1 = max(0, c - half)
                c2 = min(width - 1, c + half)

                # Define the four overlapping subwindows within [r1:r2, c1:c2]
                # top-left
                tl_r1, tl_r2 = r1, r
                tl_c1, tl_c2 = c1, c

                # top-right
                tr_r1, tr_r2 = r1, r
                tr_c1, tr_c2 = c, c2

                # bottom-left
                bl_r1, bl_r2 = r, r2
                bl_c1, bl_c2 = c1, c

                # bottom-right
                br_r1, br_r2 = r, r2
                br_c1, br_c2 = c, c2

                best_mean = v[r, c]
                best_var = np.inf

                regions = [
                    (tl_r1, tl_c1, tl_r2, tl_c2),
                    (tr_r1, tr_c1, tr_r2, tr_c2),
                    (bl_r1, bl_c1, bl_r2, bl_c2),
                    (br_r1, br_c1, br_r2, br_c2),
                ]

                for (ra, ca, rb, cb) in regions:
                    if ra > rb or ca > cb:
                        continue
                    area = (rb - ra + 1) * (cb - ca + 1)
                    s = rect_sum(S, ra, ca, rb, cb)
                    sq = rect_sum(SQ, ra, ca, rb, cb)
                    mu = s / area
                    var = (sq / area) - (mu * mu)
                    if var < best_var:
                        best_var = var
                        best_mean = mu

                out_v[r, c] = best_mean

        out_v = np.clip(out_v, 0, 255).astype(np.uint8)
        hsv_filtered = cv2.merge([h_ch, s_ch, out_v])
        self.img = cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)
        return self.img
        
if __name__ == "__main__":
    path = "bean.jpg"
    out_path = "bean_kuwahara.jpg"
    k = KuwaharaFilter(window_size=9)
    k.load_img(path)
    k.kuwahara()
    k.save_img(out_path)
    k.show_img(title="Kuwahara Filtered")