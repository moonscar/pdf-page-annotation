import sys
import os
import json
import glob

import fitz
import cv2

import numpy as np

from argparse import ArgumentParser

from functools import reduce
from skimage import measure
from datamodels import BoxClass
from typing import List, Optional, Tuple


CAPTION_LABEL_COLOR = [0, 255, 0]
CAPTION_TEXT_COLOR = [0, 0, 255]
FIGURE_BOX_COLOR = [255, 0, 0]
TABLE_BOX_COLOR = [255, 241, 0]
BACKGROUND_COLOR = [255, 255, 255]
CAPTION_OFFSET = 5


def build_args():
    parser = ArgumentParser(description='Process different version pdf, extract figures position, save page figures and build dataset.')

    parser.add_argument('target_path', type=str, help='Target path should be a dir with some paper dir, each paper dir should have black.pdf and color.pdf .')
    parser.add_argument('image_path', type=str, help='Image output path, script will save pdf page image in it.')

    parser.add_argument('--only-merge', action='store_true', help='Only execute merge ')
    parser.add_argument('--dataset-format', choices=['coco', 'label-studio'])

    args = parser.parse_args()
    return args


def im_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns a copy of image 'a' with all pixels where 'a' and 'b' are equal set to white."""
    assert (np.array_equal(np.shape(a), np.shape(b)))
    diff = a - b
    mask = np.any(diff != 0, axis=2)  # Check if any channel is different
    rgb_mask = np.transpose(np.tile(mask, (3, 1, 1)), axes=[1, 2, 0])
    diff_image = np.copy(a)
    diff_image[np.logical_not(rgb_mask)] = 255
    return diff_image


def proposal_up(full_box: BoxClass, caption_box: BoxClass) -> BoxClass:
    return BoxClass(
        x1=full_box.x1,
        y1=full_box.y1,
        x2=full_box.x2,
        y2=caption_box.y1 - CAPTION_OFFSET
    )


def proposal_down(full_box: BoxClass, caption_box: BoxClass) -> BoxClass:
    return BoxClass(
        x1=full_box.x1,
        y1=caption_box.y2 + CAPTION_OFFSET,
        x2=full_box.x2,
        y2=full_box.y2
    )


def proposal_left(full_box: BoxClass, caption_box: BoxClass) -> BoxClass:
    return BoxClass(
        x1=full_box.x1,
        y1=full_box.y1,
        x2=caption_box.x1 - CAPTION_OFFSET,
        y2=full_box.y2
    )


def proposal_right(full_box: BoxClass, caption_box: BoxClass) -> BoxClass:
    return BoxClass(
        x1=caption_box.x2 + CAPTION_OFFSET,
        y1=full_box.y1,
        x2=full_box.x2,
        y2=full_box.y2
    )


def get_figure_box(full_box: BoxClass, caption_box: BoxClass,
                   im: np.ndarray) -> Optional[BoxClass]:
    """Find the largest box inside the full figure box that doesn't overlap the caption."""
    proposals = [
        f(full_box, caption_box)
        for f in [proposal_up, proposal_down, proposal_left, proposal_right]
    ]
    proposal_areas = [p.get_area() for p in proposals]
    proposal = proposals[np.argmax(proposal_areas)]
    return proposal.crop_whitespace_edges(im)


def find_figures_and_captions(diff_im: np.ndarray, im: np.ndarray, page_num: int
    ):# -> List[Figure]:
    figures = []
    all_box_mask = (
        np.logical_or(diff_im == FIGURE_BOX_COLOR, diff_im == TABLE_BOX_COLOR)
    ).all(axis=2)
    all_caption_mask = (
        np.logical_or(
            diff_im == CAPTION_LABEL_COLOR, diff_im == CAPTION_TEXT_COLOR
        )
    ).all(axis=2)
    components = measure.label(all_box_mask)
    # Component id 0 is for background
    for component_id in np.unique(components)[1:]:
        (box_ys, box_xs) = np.where(components == component_id)
        assert (len(box_ys) > 0
               )  # It was found from np.unique so it must exist somewhere
        assert (len(box_xs) > 0)
        full_box = BoxClass(
            x1=float(min(box_xs)),
            y1=float(min(box_ys)),
            x2=float(max(box_xs) + 1),
            y2=float(max(box_ys) + 1)
        )
        caption_mask = all_caption_mask.copy()
        caption_mask[:, :round(full_box.x1)] = 0
        caption_mask[:, round(full_box.x2):] = 0
        caption_mask[:round(full_box.y1), :] = 0
        caption_mask[round(full_box.y2):, :] = 0
        (cap_ys, cap_xs) = np.where(caption_mask)
        if len(cap_ys) == 0:
            print("Ignore boxes with no captions.")
            continue  # Ignore boxes with no captions
        cap_box = BoxClass(
            x1=float(min(cap_xs) - CAPTION_OFFSET),
            y1=float(min(cap_ys) - CAPTION_OFFSET),
            x2=float(max(cap_xs) + CAPTION_OFFSET),
            y2=float(max(cap_ys) + CAPTION_OFFSET),
        )
        fig_box = get_figure_box(full_box, cap_box, im)
        if fig_box is None:
            print("fig_box is None")
            continue

        box_color = diff_im[box_ys[0], box_xs[0], :]
        if np.all(box_color == FIGURE_BOX_COLOR):
            figure_type = 'Figure'
        else:
            assert np.all(box_color == TABLE_BOX_COLOR), print(
                'Bad box color: %s' % str(box_color)
            )
            figure_type = 'Table'
        (page_height, page_width) = diff_im.shape[:2]

        print("figure_boundary", fig_box.x1, fig_box.y1, fig_box.x2, fig_box.y2)
        print("caption_boundary", cap_box.x1, cap_box.y1, cap_box.x2, cap_box.y2)
        print("figure_type", figure_type)
        print("name", '')
        print("page", page_num)
        print("caption", '')
        print("dpi", 100)
        print("page_width", page_width)
        print("page_height", page_height)
        print("\n\n")

        figures.append({
                "figure_boundary": (fig_box.x1, fig_box.y1, fig_box.x2, fig_box.y2),
                "caption_boundary": (cap_box.x1, cap_box.y1, cap_box.x2, cap_box.y2),
                "figure_type": figure_type
            })

        # figures.append(
        #     Figure(
        #         figure_boundary=fig_box,
        #         caption_boundary=cap_box,
        #         figure_type=figure_type,
        #         name='',
        #         page=page_num,
        #         caption='',
        #         dpi=settings.DEFAULT_INFERENCE_DPI,
        #         page_width=page_width,
        #         page_height=page_height
        #     )
        # )
    return figures


def compare_pdf(fileid, color_file, black_file, img_dir):
    def get_all_page_mat(filename):
        doc = fitz.open(filename)

        pages = [page for page in doc.pages()]
        page_mats = []
        mat = fitz.Matrix(1.5, 1.5)  # zoom factor 2 in each dimension

        for p in pages:
            pix = p.get_pixmap(matrix=mat)
            page_img_bytes = pix.tobytes()
            page_img_mat = np.fromstring(page_img_bytes, np.uint8)
            imageBGR = cv2.imdecode(page_img_mat, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(imageBGR , cv2.COLOR_BGR2RGB)

            page_mats.append(image)

        doc.close()
        return page_mats

    color_page_mats = get_all_page_mat(color_file)
    black_page_mats = get_all_page_mat(black_file)

    images = []
    # "images": [
    #     {
    #         "id": int,
    #         "width": int,
    #         "height": int,
    #         "file_name": str,
    #         "license": int,
    #         "flickr_url": str,
    #         "coco_url": str,
    #         "date_captured": datetime,
    #     }
    # ],

    annotations = []
    # "annotations": [
    # {
    #     "id": int,
    #     "image_id": int,
    #     "category_id": int,
    #     "segmentation": RLE or [polygon],
    #     "area": float,
    #     "bbox": [x,y,width,height],
    #     "iscrowd": 0 or 1,
    # }

    anno_id = 1
    cate_map = {
        "Figure": 5,
        "Table": 4
    }

    for page_num in range(len(color_page_mats)):
        color = color_page_mats[page_num]
        black = black_page_mats[page_num]
        diff_image = im_diff(color, black)
        zones = find_figures_and_captions(diff_image, color, 0)

        if not zones:
            continue

        filename = "{}/paper_{}-page_{}.png".format(img_dir, fileid, page_num)
        cv2.imwrite(filename, cv2.cvtColor(black, cv2.COLOR_RGB2BGR))

        height, width, _ = color.shape
        images.append({
            "id": page_num,
            "width": width,
            "height": height,
            "file_name": filename,
        })
        for area in zones:
            if area["figure_type"] == "Figure":
                trans_rect = trans_coco(area["figure_boundary"])
                bbox = trans_rect["bbox"]
                seg = trans_rect["poly"]
            elif area["figure_type"] == "Table":
                trans_rect = trans_coco(area["figure_boundary"])
                bbox = trans_rect["bbox"]
                seg = trans_rect["poly"]

            anno = {
                "id": anno_id,
                "image_id": page_num,
                "category_id": cate_map[area["figure_type"]],
                "segmentation": [seg],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            annotations.append(anno)
            anno_id += 1

    return images, annotations


def trans_coco(point_rect):
    rectangle = [int(i) for i in point_rect]

    start_point = rectangle[0], rectangle[1]
    width = rectangle[2] - rectangle[0]
    height = rectangle[3] - rectangle[1]
    second_point = rectangle[1] + width, rectangle[0]
    third_point = rectangle[1] + width, rectangle[0] + height
    fourth_point = rectangle[1], rectangle[0] + height

    point_path = [start_point, second_point, third_point, fourth_point]
    poly_list = reduce(lambda x,y:x+y, point_path)

    return {
        "poly": poly_list,
        "bbox": [start_point[0], start_point[1], width, height]
    }


def main(args):
    target_path = args.target_path
    image_path = args.image_path

    if not args.only_merge:
        for path in glob.glob("{}/*".format(target_path)):
            fileid = os.path.basename(path)
            color_file = os.path.join(path, "color.pdf")
            black_file = os.path.join(path, "black.pdf")

            if not os.path.exists(color_file):
                continue

            images, annotations = compare_pdf(fileid, color_file, black_file, image_path)
            if annotations:
                coco_dataset = {
                    "info": {},
                    "images": images,
                    "annotations": annotations,
                }

                with open("{}/data.json".format(path), "w+") as w:
                    json.dump(coco_dataset, w)

    if args.dataset_format == "coco":
        merge_in_coco("{}/*/data.json".format(target_path), image_path)
    elif args.dataset_format == "label-studio":
        merge_in_ls_json("{}/*/data.json".format(target_path), image_path)
    else:
        print("Your dataset format is invalid, there will be no final json.")


def merge_in_coco(file_pattern, image_path):
    start_file_id = 1
    start_anno_id = 1

    coco_dataset = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [
        {
            "supercategory": "",
            "id": 1,
            "name": "OtherRegion"
        },
        {
            "supercategory": "",
            "id": 2,
            "name": "SeparatorRegion"
        },
        {
            "supercategory": "",
            "id": 3,
            "name": "MathsRegion"
        },
        {
            "supercategory": "",
            "id": 4,
            "name": "TableRegion"
        },
        {
            "supercategory": "",
            "id": 5,
            "name": "ImageRegion"
        }]
    }

    for f in glob.glob(file_pattern):
        temp_file_map = {}

        with open(f) as f:
            file_json = json.load(f)

        for i, image in enumerate(file_json["images"], start=start_file_id):
            temp_file_map[image["id"]] = i
            image["id"] = i
            rel_path = image["file_name"].replace("/nas2/hyy/layout-datasets/deepfigures-caiyun/", "")
            image["file_name"] = rel_path
            coco_dataset["images"].append(image)

        for i, anno in enumerate(file_json["annotations"], start=start_anno_id):
            anno["id"] = i
            anno["image_id"] = temp_file_map[anno["image_id"]]
            coco_dataset["annotations"].append(anno)

        start_file_id += len(file_json["images"])
        start_anno_id += len(file_json["annotations"])

    with open("coco_way.json", "w+") as w:
        json.dump(coco_dataset, w)

    return None


def merge_in_ls_json(file_pattern, image_path):
    result = []

    for f in glob.glob(file_pattern):
        temp_file_map = {}

        with open(f) as f:
            file_json = json.load(f)

        for i, image in enumerate(file_json["images"]):
            rel_path = image["file_name"].replace("/nas2/hyy/layout-datasets/", "")
            image["file_name"] = "http://durian.in.caiyunapp.com:8081/{}".format(rel_path)
            temp_file_map[image["id"]] = image

        for i, anno in enumerate(file_json["annotations"]):
            image = temp_file_map[anno["image_id"]]
            image_width = image["width"]
            image_height = image["height"]
            bbox = anno["bbox"]

            label = {
                "data": {
                    "image": image["file_name"]
                },
                "predictions": [{
                    "result": [
                    {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "source": "$image",
                        "original_height": image_height,
                        "original_width": image_width,
                        "value": {
                            "x": 100 * bbox[0] / image_width,
                            "y": 100 * bbox[1] / image_height,
                            "width": 100 * bbox[2] / image_width,
                            "height": 100 * bbox[3] / image_height,
                            "rectanglelabels": [
                                "TableRegion"
                            ],
                            "rotation": 0,
                        }
                    }]
                }],
            }

            result.append(label)

    with open("label-studio_way.json", "w+") as w:
        json.dump(result, w)


if __name__ == '__main__':
    args = build_args()
    main(args)
