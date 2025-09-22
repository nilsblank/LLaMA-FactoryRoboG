

import copy
import os
import json
import pickle
from multiprocessing import Pool, cpu_count
import av
import numpy as np
from tqdm import tqdm
from PIL import Image

TEMPLATE = {
        "messages": [
            {
                "content": '''<image><video> Given the video, determine the object the human interacted with. Output the label and location of the interacted object in the provided frame in JSON format.''',
                "role": "user"
            },
            {
                "content": "",
                "role": "assistant"
            },
        ],
        "videos": [
        ],
        "images": []
    }


bbox_format = '''```json
[
{"bbox_2d": BBOX_COORDS, "label": "OBJECT_NAME"}
]
```
'''




import math

# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]


def create_hd_epic_dataset(annotations_file,dataset_name, out_dir):


    eval_kitchen = "P07"

    train_json_out_path = os.path.join(out_dir, f"{dataset_name}_first_frame_train.json")
    eval_json_out_path = os.path.join(out_dir, f"{dataset_name}_first_frame_eval.json")
    debug_path = os.path.join(out_dir, f"{dataset_name}_first_frame_debug.json")


    eval_data = []
    train_data = []

    os.makedirs(out_dir, exist_ok=True)
    with open(annotations_file, "rb") as f:
        annotations = json.load(f)
        
        
    per_kitchen_counts = {}

    eval_size = 128
    
    eval_indices = np.random.choice(list(annotations.keys()), size=eval_size, replace=False)
    idx = 0
    for track_id, video_data in tqdm(annotations.items(), desc="Processing videos"):
        

        cur_data = copy.deepcopy(TEMPLATE)
        cur_data["videos"] =  [ "/home/blank" + video_data["video_path"]]
        first_frame_save_dir = cur_data["videos"][0].replace(".mp4", "_first_frame.jpg")
        cur_data["images"] = [first_frame_save_dir]
        
        #check if video can be opened
        try:
            container = av.open(cur_data["videos"][0], "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            #get first 
            if video_stream is None:
                print(f"No video stream found in {cur_data['videos'][0]}")
                continue
            for frame in container.decode(container.streams.video[0]):
                first_frame = frame
                break


            frame = Image.fromarray(first_frame.to_ndarray(format="rgb24"))
            frame.save(first_frame_save_dir)
        
            
        except Exception as e:
            print(f"Error opening video {cur_data['videos'][0]}: {e}")
            continue
        initial_position = video_data["bboxes"][0]

        initial_position = [int(value) for value in initial_position]
        initial_position = convert_to_qwen25vl_format(initial_position, 1408, 1408)

        obj_name = video_data["name"]
        label = bbox_format.replace("BBOX_COORDS", str(initial_position)).replace("OBJECT_NAME", obj_name)
        cur_data["messages"][1]["content"] = label

        kitchen = video_data["video_path"].split("/")[5]
        
        if kitchen not in per_kitchen_counts:
            per_kitchen_counts[kitchen] = 0
        per_kitchen_counts[kitchen] += 1
        
        # if eval_kitchen in video_data["video_path"]:
        #     eval_data.append(cur_data)
        # else:
        #     train_data.append(cur_data)
        
        if track_id in eval_indices:
            eval_data.append(cur_data)
        else:
            train_data.append(cur_data)

    with open(train_json_out_path, "w") as f:
        json.dump(train_data, f, indent=4)
        print(f"Train Size: {len(train_data)}")
    with open(eval_json_out_path, "w") as f:
        json.dump(eval_data, f, indent=4)
        print(f"Eval Size: {len(eval_data)}")

    
        
    
    with open(debug_path, "w") as f:
        json.dump(np.random.choice(train_data, size=32, replace=False).tolist(), f, indent=4)

if __name__ == "__main__":
    annotations_file = "/home/blank/data/hd-epic/HD-EPIC/processed_annotations_boxes.json"
    dataset_name = "hd_epic"
    out_dir = "/home/blank/data/LLAMA-factory/"

    create_hd_epic_dataset(annotations_file, dataset_name, out_dir)


        






   




