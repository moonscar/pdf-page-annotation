from matplotlib import patches
from typing import List, Optional, Tuple
import numpy as np

BACKGROUND_COLOR = 255


class BoxClass():
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @staticmethod
    def from_tuple(t: Tuple[float, float, float, float]) -> 'BoxClass':
        return BoxClass(x1=t[0], y1=t[1], x2=t[2], y2=t[3])

    def get_width(self) -> float:
        return self.x2 - self.x1

    def get_height(self) -> float:
        return self.y2 - self.y1

    def get_plot_box(
        self, color: str='red', fill: bool=False, **kwargs
    ) -> patches.Rectangle:
        """Return a rectangle patch for plotting"""
        return patches.Rectangle(
            (self.x1, self.y1),
            self.get_width(),
            self.get_height(),
            edgecolor=color,
            fill=fill,
            **kwargs
        )

    def get_area(self) -> float:
        width = self.get_width()
        height = self.get_height()
        if width <= 0 or height <= 0:
            return 0
        else:
            return width * height

    def rescale(self, ratio: float) -> 'BoxClass':
        return BoxClass(
            x1=self.x1 * ratio,
            y1=self.y1 * ratio,
            x2=self.x2 * ratio,
            y2=self.y2 * ratio
        )

    # def resize_by_page(
    #     self, cur_page_size: ImageSize, target_page_size: ImageSize
    # ):
    #     (orig_h, orig_w) = cur_page_size[:2]
    #     (target_h, target_w) = target_page_size[:2]
    #     height_scale = target_h / orig_h
    #     width_scale = target_w / orig_w
    #     return BoxClass(
    #         x1=self.x1 * width_scale,
    #         y1=self.y1 * height_scale,
    #         x2=self.x2 * width_scale,
    #         y2=self.y2 * height_scale
    #     )

    def get_rounded(self):# -> IntBox:
        return (
            int(round(self.x1)), int(round(self.y1)), int(round(self.x2)),
            int(round(self.y2))
        )

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Return image cropped to the portion contained in box."""
        (x1, y1, x2, y2) = self.get_rounded()
        return image[y1:y2, x1:x2]

    def crop_whitespace_edges(self, im: np.ndarray) -> Optional['BoxClass']:
        (rounded_x1, rounded_y1, rounded_x2, rounded_y2) = self.get_rounded()
        white_im = im.copy()
        white_im[:, :rounded_x1] = BACKGROUND_COLOR
        white_im[:, rounded_x2:] = BACKGROUND_COLOR
        white_im[:rounded_y1, :] = BACKGROUND_COLOR
        white_im[rounded_y2:, :] = BACKGROUND_COLOR
        is_white = (white_im == BACKGROUND_COLOR).all(axis=2)
        nonwhite_columns = np.where(is_white.all(axis=0) != 1)[0]
        nonwhite_rows = np.where(is_white.all(axis=1) != 1)[0]
        if len(nonwhite_columns) == 0 or len(nonwhite_rows) == 0:
            return None
        x1 = min(nonwhite_columns)
        x2 = max(nonwhite_columns) + 1
        y1 = min(nonwhite_rows)
        y2 = max(nonwhite_rows) + 1
        assert x1 >= rounded_x1, 'ERROR:  x1:%d box[0]:%d' % (x1, rounded_x1)
        assert y1 >= rounded_y1, 'ERROR:  y1:%d box[1]:%d' % (y1, rounded_y1)
        assert x2 <= rounded_x2, 'ERROR:  x2:%d box[2]:%d' % (x2, rounded_x2)
        assert y2 <= rounded_y2, 'ERROR:  y2:%d box[3]:%d' % (y2, rounded_y2)
        # np.where returns np.int64, cast back to python types
        return BoxClass(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

    def distance_to_other(self, other: 'BoxClass') -> float:
        x_distance = max([0, self.x1 - other.x2, other.x1 - self.x2])
        y_distance = max([0, self.y1 - other.y2, other.y1 - self.y2])
        return np.linalg.norm([x_distance, y_distance], 2)

    def intersection(self, other: 'BoxClass') -> float:
        intersection = BoxClass(
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
            x2=min(self.x2, other.x2),
            y2=min(self.y2, other.y2)
        )
        if intersection.x2 >= intersection.x1 and intersection.y2 >= intersection.y1:
            return intersection.get_area()
        else:
            return 0

    def iou(self, other: 'BoxClass') -> float:
        intersection = self.intersection(other)
        union = self.get_area() + other.get_area() - intersection
        if union == 0:
            return 0
        else:
            return intersection / union

    def contains_box(self, other: 'BoxClass', overlap_threshold=.5) -> bool:
        if other.get_area() == 0:
            return False
        else:
            return self.intersection(other
                                    ) / other.get_area() >= overlap_threshold

    def expand_box(self, amount: float) -> 'BoxClass':
        return BoxClass(
            x1=self.x1 - amount,
            y1=self.y1 - amount,
            x2=self.x2 + amount,
            y2=self.y2 + amount,
        )

    def crop_to_page(self, page_shape) -> 'BoxClass':
        page_height, page_width = page_shape[:2]
        return BoxClass(
            x1=max(self.x1, 0),
            y1=max(self.y1, 0),
            x2=min(self.x2, page_width),
            y2=min(self.y2, page_height),
        )
