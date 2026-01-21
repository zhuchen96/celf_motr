import os
import json
from glob import glob
import configparser

"""
import debugpy

debugpy.listen(("0.0.0.0", 5678))  # listen for the debugger
print("⏳ Waiting for debugger attach...")
debugpy.wait_for_client()  # pause until debugger is attached
debugpy.breakpoint()  # this acts like a manual breakpoint
"""

DANCETRACK_ROOT = '/images/SegmentationDistillation/data/DanceTrack/val'
OUT_JSON = '/images/SegmentationDistillation/data/DanceTrack/val_json/val.json'

video_dirs = sorted(glob(os.path.join(DANCETRACK_ROOT, 'dancetrack*/')))

images = []
annotations = []
categories = [{'id': 1, 'name': 'person'}]

image_id_counter = 1
ann_id_counter = 1

for video_dir in video_dirs:
    video_name = os.path.basename(os.path.normpath(video_dir))
    gt_file = os.path.join(video_dir, 'gt', 'gt.txt')
    seqinfo_file = os.path.join(video_dir, 'seqinfo.ini')

    # Parse image dimensions from seqinfo.ini
    config = configparser.ConfigParser()
    config.read(seqinfo_file)

    width = int(config['Sequence']['imWidth'])
    height = int(config['Sequence']['imHeight'])

    frame_to_image_id = {}

    with open(gt_file, 'r') as f:
        for line in f:
            frame, pid, x, y, w, h, _, _, _ = map(float, line.strip().split(','))
            frame = int(frame)
            pid = int(pid)

            image_name = f"{frame:08d}.jpg"
            file_name = f"{video_name}/img1/{image_name}"

            if frame not in frame_to_image_id:
                # New image/frame
                frame_to_image_id[frame] = image_id_counter
                images.append({
                    'id': image_id_counter,
                    'file_name': file_name,
                    'frame_id': frame,
                    'height': height,
                    'width': width
                })
                image_id_counter += 1

            annotations.append({
                'id': ann_id_counter,
                'image_id': frame_to_image_id[frame],
                'category_id': 1,
                'bbox': [x, y, w, h],
                'iscrowd': 0,
                'area': w * h
            })
            ann_id_counter += 1

coco = {
    'images': images,
    'annotations': annotations,
    'categories': categories
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(coco, f)

print(f"Saved COCO annotations to {OUT_JSON}")
