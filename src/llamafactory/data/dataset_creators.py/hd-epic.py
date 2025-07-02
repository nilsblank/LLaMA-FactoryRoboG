#%%
import json
import os
import pickle
import pandas as pd
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

video_dir = "/data/hd-epic/HD-EPIC/Videos"

annotations_file = "/data/hd-epic/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
bbox_movement_assoc = "/data/hd-epic/hd-epic-annotations/scene-and-object-movements/assoc_info.json"
mask_file = "/data/hd-epic/hd-epic-annotations/scene-and-object-movements/mask_info.json"


with open(annotations_file, "rb") as f:
    annotations = pickle.load(f)

with open(bbox_movement_assoc, "r") as f:
    bbox_movement_assoc = json.load(f)
with open(mask_file, "r") as f:
    mask_info = json.load(f)

def process_track(item, annotations_df, mask_info_dict, video_base_dir):



    video_id = item["video_id"]
    assoc_name = item["assoc_name"]
    track = item["track"]

    kitchen_dir = video_id.split("-")[0]
    video_path = f"{video_base_dir}/{kitchen_dir}/{video_id}.mp4"


    out_dir = f"/data/hd-epic/HD-EPIC/processed_videos/{kitchen_dir}/{track['track_id']}.mp4"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    
    cut_video = True
    if os.path.exists(out_dir):
        cut_video = False

    if not os.path.exists(video_path):
        return None

    cur_df = annotations_df[annotations_df["video_id"] == video_id]
    if cur_df.empty:
        return None

    mask_ids = track["masks"]
    start_time, end_time = track["time_segment"][0], track["time_segment"][1]

    start_time -= 1
    end_time += 1

    filtered_df = cur_df[(cur_df["start_timestamp"] >= start_time) & (cur_df["end_timestamp"] <= end_time)]

    if filtered_df.empty:
        return None

    lang_instructions = filtered_df["narration"].tolist()
    concat_lang = " ".join(lang_instructions)

    if not mask_ids:
        return None
        
    try:
        bboxes = [mask_info_dict[video_id][mask_id]["bbox"] for mask_id in mask_ids]
        frame_numbers = [mask_info_dict[video_id][mask_id]["frame_number"] for mask_id in mask_ids]
    except KeyError:
        # Skip if mask_id not found for this video_id
        return None

    if not frame_numbers:
        return None
    
    n_frames = frame_numbers[-1] - frame_numbers[0] + 1
    print(n_frames)
    if  n_frames > 250 or n_frames < 20:
        print(f"Skipping track {track['track_id']} for video {video_id} due to excessive frame range.")
        return None     

    frame_indices = list(range(frame_numbers[0], frame_numbers[-1] + 1))


    if cut_video:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(out_dir, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
            writer.write(frame)
        
        cap.release()
        writer.release()

    return {
        f"{track['track_id']}": {
            "video_path": out_dir,
            "bboxes": bboxes,
            "frame_numbers": frame_numbers,
            "name": assoc_name,
            "lang": concat_lang,
                            }
    }

#%%
if __name__ == '__main__':
    track_items = []
    for video_id, data in bbox_movement_assoc.items():
        for assoc_id, assoc_data in data.items():
            for track in assoc_data["tracks"]:
                track_items.append({
                    "video_id": video_id,
                    "assoc_name": assoc_data["name"],
                    "track": track
                })
    track_items = track_items
    worker_func = partial(process_track, annotations_df=annotations, mask_info_dict=mask_info, video_base_dir=video_dir)
    
    annotations_boxes = {}
    
    num_processes = cpu_count() // 2
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(worker_func, track_items), total=len(track_items), desc="Processing tracks"))

    for res_dict in results:
        if res_dict:
            annotations_boxes.update(res_dict)

    out_path = "/data/hd-epic/HD-EPIC/processed_annotations_boxes.json"
    with open(out_path, "w") as f:
        json.dump(annotations_boxes, f, indent=4)


    
# %%
