import json
import os
from pycocotools.coco import COCO

def convert_coco_to_yolo(coco_json_path, image_dir, output_dir):
    # Load COCO annotations
    coco = COCO(coco_json_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get category mapping (COCO category_id to YOLO class_id)
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_yolo = {cat['id']: idx for idx, cat in enumerate(cats)}

    # Process each image
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_file = img_info['file_name']
        img_width, img_height = img_info['width'], img_info['height']

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Create YOLO annotation file
        yolo_file = os.path.join(output_dir, img_file.replace('.jpg', '.txt'))
        with open(yolo_file, 'w') as f:
            for ann in anns:
                bbox = ann['bbox']  # [x_min, y_min, width, height]
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                class_id = cat_id_to_yolo[ann['category_id']]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Paths
splits = ['train', 'val', 'test']
for split in splits:
    coco_json = f'yolo/dataset/annotations/{split}.json'
    img_dir = f'yolo/dataset/images/{split}'
    output_dir = f'yolo/dataset/labels/{split}'
    convert_coco_to_yolo(coco_json, img_dir, output_dir)