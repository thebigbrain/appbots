import json
import os.path
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from appbots.core.utils import get_coco_dir, get_yolo_path, write_lines
from appbots.core.yolo.utils import make_yolo_dirs


def coco_box2yolo_label(box: list[float], w: float, h: float):
    box = np.array(box, dtype=np.float64)
    box[:2] += box[2:] / 2  # xy top-left corner to center
    box[[0, 2]] /= w  # normalize x
    box[[1, 3]] /= h  # normalize y

    return box


def convert_coco_json(coco_json_dir, yolo_name="coco", use_segments=False):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    save_dir = make_yolo_dirs(get_yolo_path(yolo_name))  # output directory

    # Import json
    for json_file in sorted(Path(coco_json_dir).resolve().glob("*.json")):
        with open(json_file, mode='r', encoding="utf-8") as jf:
            data = json.load(jf)

        label_train_dir = Path(save_dir) / "labels"  # folder name
        label_train_dir.mkdir(exist_ok=True)
        image_train_dir = Path(save_dir) / "images"
        image_train_dir.mkdir(exist_ok=True)

        # Create image dict
        images = {"{:g}".format(x["id"]): x for x in data["images"]}

        # Create image-annotations dict
        img_to_anns = defaultdict(list)
        for ann in data["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        # Write label file
        for img_id, anns in tqdm(img_to_anns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:g}"]
            h, w, img_file_name = img["height"], img["width"], img["file_name"]

            img_base_name = os.path.basename(img_file_name)
            shutil.copy(img_file_name, image_train_dir / img_base_name)

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue

                # The COCO box format is [top left x, top left y, width, height]
                box = coco_box2yolo_label(ann["bbox"], w, h)
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann["category_id"]   # class

                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    if len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            lines = []
            file_name = (label_train_dir / img_base_name).with_suffix(".txt")
            for i in range(len(bboxes)):
                line = (*(segments[i] if use_segments else bboxes[i]),)  # cls, box or segments
                line_format = ("%g " * len(line)).rstrip()
                lines.append(line_format % line)

            write_lines(file_name, lines)


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).

    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]: idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


if __name__ == "__main__":
    convert_coco_json(
        get_coco_dir(),  # directory with *.json
    )
    # zip results
    # os.system('zip -r ../coco.zip ../coco')
