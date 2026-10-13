import os
import cv2
import shutil

def extract_frames(video_path, output_dir, frame_skip=30, max_width=1280):
    """
    Extracts and downscales frames from a high-fps, high-resolution video.
    
    :param video_path: Path to the input video file.
    :param output_dir: Directory where extracted frames will be saved.
    :param frame_skip: Save every N-th frame (default: 30 for 1 frame/sec at 60fps).
    :param max_width: Resize frame width to keep processing fast (default: 1280).
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Deletes the old folder and all its contents
    
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing: {video_path} | FPS: {fps:.2f} | Total Frames: {total_frames}")

    frame_idx = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every N-th frame
        if frame_idx % frame_skip == 0:
            # Resize 4K down to a manageable width (preserving aspect ratio)
            if max_width and frame.shape[1] > max_width:
                h, w = frame.shape[:2]
                aspect_ratio = h / w
                new_width = max_width
                new_height = int(max_width * aspect_ratio)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            output_filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(output_filename, frame)
            saved_count += 1
            
        frame_idx += 1

    cap.release()
    print(f"-> Saved {saved_count} downscaled frames to {output_dir}\n")

if __name__ == "__main__":
    videos = {
        "scene_v1": "data/raw_videos/scene_v1.mp4",
        "scene_v2": "data/raw_videos/scene_v2.mp4"
    }
    
    for scene_name, path in videos.items():
        output_folder = os.path.join("data", "processed_frames", scene_name)
        # Using frame_skip=30 (1 frame per sec for 60fps) and max_width=1280 (720p/HD)
        extract_frames(path, output_folder, frame_skip=30, max_width=1280)