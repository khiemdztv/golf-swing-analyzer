import cv2
import mediapipe as mp
import json
import os
from tqdm import tqdm

mp_pose = mp.solutions.pose

def extract_landmarks(video_path, visualize=False):
    """Extract landmarks với option visualize để kiểm tra"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    # Tăng confidence để detection chính xác hơn
    pose = mp_pose.Pose(
        model_complexity=2,  # Dùng model phức tạp nhất
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    frames = []
    frame_count = 0
    detected_count = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer nếu visualize
    if visualize:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = video_path.replace('.mp4', '_visualized.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames, desc="Extracting frames")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            detected_count += 1
            pts = []
            for lm in res.pose_landmarks.landmark:
                pts.append([lm.x, lm.y, lm.z])
            frames.append(pts)
            
            # Visualize nếu cần
            if visualize:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
        
        if visualize:
            out.write(frame)
        
        pbar.update(1)

    pbar.close()
    cap.release()
    
    if visualize:
        out.release()
        print(f"   📹 Saved visualization: {output_path}")
    
    detection_rate = (detected_count / frame_count * 100) if frame_count > 0 else 0
    print(f"   📊 Detection rate: {detection_rate:.1f}% ({detected_count}/{frame_count} frames)")
    
    # Cảnh báo nếu detection rate thấp
    if detection_rate < 70:
        print(f"   ⚠️  Low detection rate! Video quality might be poor.")
    
    return frames


def process_folder(folder, output_folder=None, visualize=False):
    """Process tất cả video trong folder"""
    
    if output_folder is None:
        output_folder = folder
    
    # Tạo folder output nếu chưa có
    os.makedirs(output_folder, exist_ok=True)
    
    video_files = [f for f in os.listdir(folder) 
                   if f.lower().endswith((".mp4", ".mov", ".avi"))]
    
    if not video_files:
        print(f"⚠️  No video files found in {folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(video_files)} videos in {folder}")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {file}")
        print("-" * 60)
        
        video_path = os.path.join(folder, file)
        
        try:
            data = extract_landmarks(video_path, visualize=visualize)
            
            if data is None or len(data) == 0:
                print(f"   ❌ Failed to extract landmarks")
                results.append({
                    "file": file,
                    "status": "failed",
                    "frames": 0
                })
                continue
            
            # Save JSON
            output_name = os.path.splitext(file)[0] + ".json"
            output_path = os.path.join(output_folder, output_name)
            
            with open(output_path, "w") as f:
                json.dump(data, f)
            
            print(f"   ✅ Saved {len(data)} frames to: {output_name}")
            
            results.append({
                "file": file,
                "status": "success",
                "frames": len(data)
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                "file": file,
                "status": "error",
                "frames": 0,
                "error": str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    success = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - success
    total_frames = sum(r["frames"] for r in results)
    
    print(f"✅ Success: {success}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"📊 Total frames extracted: {total_frames}")
    
    if failed > 0:
        print(f"\n⚠️  Failed videos:")
        for r in results:
            if r["status"] != "success":
                print(f"   - {r['file']}: {r.get('error', 'Unknown error')}")


def batch_process_with_structure(base_folder, visualize=False):
    """Process theo cấu trúc folder sideview/backview"""
    
    side_folder = os.path.join(base_folder, "sideview")
    back_folder = os.path.join(base_folder, "backview")
    
    print("🏌️ GOLF SWING POSE EXTRACTION")
    print("="*60)
    
    # Process sideview
    if os.path.exists(side_folder):
        print(f"\n📂 Processing SIDEVIEW folder...")
        process_folder(side_folder, visualize=visualize)
    else:
        print(f"\n⚠️  Sideview folder not found: {side_folder}")
    
    # Process backview
    if os.path.exists(back_folder):
        print(f"\n📂 Processing BACKVIEW folder...")
        process_folder(back_folder, visualize=visualize)
    else:
        print(f"\n⚠️  Backview folder not found: {back_folder}")
    
    print(f"\n{'='*60}")
    print("🎉 EXTRACTION COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Đường dẫn folder gốc chứa sideview và backview
    # Thay đổi đường dẫn này theo máy của bạn
    base_path = r"D:\Documents\Data Storm\video vdv pro"
    
    # Set visualize=True để tạo video có vẽ skeleton (để kiểm tra)
    # Set visualize=False để chạy nhanh hơn
    visualize = False
    
    # Nếu muốn process từng folder riêng
    # process_folder(r"path/to/your/folder", visualize=False)
    
    # Hoặc process theo cấu trúc
    batch_process_with_structure(base_path, visualize=visualize)