import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import math
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# ---------------------------------------------------------------------------
# Self-contained core logic (avoids 'No module named utils' on Streamlit Cloud)
# ---------------------------------------------------------------------------
class RunningAnalyzer:
    def __init__(self):
        self.positions, self.total_distance = {}, {}
        self.current_speed, self.max_speed = {}, {}

    def update(self, tid, box):
        center = ((box[0]+box[2])//2, (box[1]+box[3])//2)
        if tid not in self.positions:
            self.positions[tid] = [center]
            self.total_distance[tid] = 0.0
            self.current_speed[tid] = 0.0
            self.max_speed[tid] = 0.0
            return
        last = self.positions[tid][-1]
        dist = math.hypot(center[0]-last[0], center[1]-last[1])
        self.total_distance[tid] += dist
        self.current_speed[tid] = dist
        if dist > self.max_speed[tid]: self.max_speed[tid] = dist
        self.positions[tid].append(center)
        if len(self.positions[tid]) > 300: self.positions[tid].pop(0)

    def get_speed(self, tid): return self.current_speed.get(tid, 0.0)
    def get_distance(self, tid): return self.total_distance.get(tid, 0.0)
    def get_max_speed(self, tid): return self.max_speed.get(tid, 0.0)
    def get_stats(self, tid): return {"speed": self.get_speed(tid), "distance": self.get_distance(tid), "max_speed": self.get_max_speed(tid)}

class FootballIQAnalyzer:
    def __init__(self): self.player_history = {}
    def update(self, tid, center):
        if tid not in self.player_history: self.player_history[tid] = []
        self.player_history[tid].append(center)
        if len(self.player_history[tid]) > 300: self.player_history[tid].pop(0)
    def total_distance(self, tid):
        pts = self.player_history.get(tid, [])
        return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]) for i in range(1, len(pts))) if len(pts) > 1 else 0
    def average_speed(self, tid):
        pts = self.player_history.get(tid, [])
        return self.total_distance(tid) / len(pts) if len(pts) > 1 else 0
    def sprint_count(self, tid, threshold=15):
        history = self.player_history.get(tid, [])
        return sum(1 for i in range(1, len(history)) if math.hypot(history[i][0]-history[i-1][0], history[i][1]-history[i-1][1]) > threshold)
    def work_rate(self, tid):
        d = self.total_distance(tid)
        return "Low" if d < 500 else ("Medium" if d < 1500 else "High")
    def direction_changes(self, tid):
        pts = self.player_history.get(tid, [])
        changes = 0
        for i in range(2, len(pts)):
            dx1, dy1 = pts[i-1][0]-pts[i-2][0], pts[i-1][1]-pts[i-2][1]
            dx2, dy2 = pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]
            m1, m2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
            if m1 == 0 or m2 == 0: continue
            angle = math.degrees(math.acos(max(-1, min(1, (dx1*dx2+dy1*dy2)/(m1*m2)))))
            if angle > 45: changes += 1
        return changes
    def movement_score(self, tid):
        d = self.total_distance(tid); t = self.direction_changes(tid); s = self.sprint_count(tid)
        return min(100, round(d*0.02 + t*2 + s*5))
    def stats(self, tid):
        return {"distance": round(self.total_distance(tid), 1), "avg_speed": round(self.average_speed(tid), 2),
                "sprints": self.sprint_count(tid), "direction_changes": self.direction_changes(tid),
                "work_rate": self.work_rate(tid), "movement_score": self.movement_score(tid)}

class HeatmapAnalyzer:
    def __init__(self): self.canvas = None
    def initialize(self, frame):
        if self.canvas is not None: return
        h, w = frame.shape[:2]
        self.canvas = np.zeros((h, w), dtype=np.float32)
    def update(self, x, y):
        if self.canvas is not None: cv2.circle(self.canvas, (int(x), int(y)), 20, 1, -1)
    def overlay(self, frame, alpha=0.45):
        if self.canvas is None: return frame
        norm = cv2.normalize(self.canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1-alpha, heat, alpha, 0)
    def clear(self):
        if self.canvas is not None: self.canvas.fill(0)

class PoseAnalyzer:
    def __init__(self, model_path="weights/yolo11n-pose.pt"):
        self.model = YOLO(model_path)
    def calculate_angle(self, a, b, c):
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return abs(ang) if abs(ang) <= 180 else 360-abs(ang)
    def analyze(self, frame, box):
        x1, y1, x2, y2 = box; h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return frame, None
        results = self.model.predict(roi, verbose=False)
        if not results or not results[0].keypoints or len(results[0].keypoints.xy[0]) < 17: return frame, None
        kp = results[0].keypoints.xy[0].cpu().numpy()
        lk = self.calculate_angle(kp[11], kp[13], kp[15])
        rk = self.calculate_angle(kp[12], kp[14], kp[16])
        lh = self.calculate_angle(kp[5], kp[11], kp[13])
        rh = self.calculate_angle(kp[6], kp[12], kp[14])
        for pt in kp:
            if pt[0] != 0: cv2.circle(roi, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        return frame, {"left_knee": lk, "right_knee": rk, "posture": "upright" if (lh+rh)/2 > 120 else "leaning"}

class VideoProcessor:
    def __init__(self):
        self.tracker = YOLO("weights/yolo11n.pt")
        self.running = RunningAnalyzer()
        self.heatmap = HeatmapAnalyzer()
        self.iq = FootballIQAnalyzer()
        self.pose = PoseAnalyzer()
        self.selected_player = None
        self.id_map = {}
        self.next_id = 1

    def process(self, frame):
        self.heatmap.initialize(frame)
        results = self.tracker.track(frame, persist=True, tracker="botsort.yaml", conf=0.3, iou=0.45, classes=[0], imgsz=1280, verbose=False)[0]
        output = frame.copy()
        if results.boxes is None: return output, None
        feedback = None
        for box in results.boxes:
            if box.id is None: continue
            internal = int(box.id[0])
            if internal not in self.id_map: self.id_map[internal] = self.next_id; self.next_id += 1
            did = self.id_map[internal]
            if int(box.cls[0]) != 0: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0]); cx, cy = (x1+x2)//2, (y1+y2)//2
            self.running.update(did, (x1, y1, x2, y2)); self.iq.update(did, (cx, cy))
            if self.selected_player == did:
                self.heatmap.update(cx, cy)
                pf, pd = self.pose.analyze(output, (x1, y1, x2, y2)); output = pf
                feedback = self.coach_feedback(did, pd)
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
            else:
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            s = self.running.get_stats(did)
            cv2.putText(output, f"ID:{did} SPD:{s['speed']:.1f} DST:{s['distance']:.1f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if self.selected_player != did else (0, 0, 255), 2)
        if self.selected_player is not None: output = self.heatmap.overlay(output)
        return output, feedback

    def coach_feedback(self, tid, pose):
        if pose is None: return "Analyzing body alignment..."
        s = self.running.get_stats(tid); sp = s['speed']; fb, rx = "", ""
        ak = (pose['left_knee']+pose['right_knee'])/2
        if ak < 140 and sp > 5: fb = "High-efficiency knee drive. "
        elif ak > 160 and sp > 5: fb = "Stiff gait. "; rx = "Rectification: A-skips + Plyometric bounds."
        if pose['posture']=='leaning' and sp < 3: fb+="Excessive lean. "; rx = "Rectification: Core stability (planks)."
        elif pose['posture']=='upright' and sp > 8: fb+="Optimal posture. "
        if sp > 12: fb+="Elite speed. "
        elif sp < 2 and self.iq.get_distance(tid) > 100: fb+="Low intensity. "; rx = "Rectification: HIIT training."
        if not fb: fb = "Steady movement."
        return f"Coach: {fb} {rx}".strip()[:100]

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FootballAI Pro Web", layout="wide")
st.title("⚽ FootballAI Pro - Elite Player Analysis")
st.markdown("Professional biometric tracking and sports-science coaching.")

# Input method: URL for large files or direct upload for smaller files
input_method = st.radio("Select video input method:", ["Enter Video URL (supports up to 4GB)", "Upload video (max 200MB)"])

video_path = None
url = None

if input_method.startswith("Enter Video URL"):
    url = st.text_input("Paste direct video URL (Google Drive, Dropbox, etc.):")
    if url and st.button("Download and Analyze"):
        with st.spinner("Downloading video from URL..."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                tfile.write(chunk)
            tfile.close()
            video_path = tfile.name
            st.success("Download complete. Starting analysis...")
else:
    uploaded_file = st.file_uploader("Upload Football Match Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False); tfile.write(uploaded_file.read())
        video_path = tfile.name

if video_path is not None:
    processor = VideoProcessor()

    # Detect players from first frame
    cap = cv2.VideoCapture(video_path); ret, frame = cap.read(); cap.release()
    if ret:
        processor.process(frame)
        players = list(processor.id_map.values())
        if players:
            sel = st.selectbox("Select Player to Analyze", players)
            processor.selected_player = sel
            if st.button("Start Professional Analysis"):
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                frames = 0; last_fb = ""
                with st.spinner("Applying Sports Science Analysis..."):
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        pf, fb = processor.process(frame); out.write(pf)
                        if fb: last_fb = fb
                        frames += 1
                cap.release(); out.release()
                st.success(f"Done! {frames} frames analyzed.")
                with open(out_path, 'rb') as f: st.video(f.read())

                st.subheader("📋 Coaching Summary")
                st.info(last_fb)
                stats = processor.running.get_stats(sel)
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Speed", f"{stats['speed']:.2f}")
                c2.metric("Total Distance", f"{stats['distance']:.2f}")
                c3.metric("Max Speed", f"{processor.running.get_max_speed(sel):.2f}")

                iq_stats = processor.iq.stats(sel)
                df = pd.DataFrame([{"PlayerID": sel, **stats, **iq_stats}])
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Report (CSV)", csv, "player_report.csv", "text/csv")
        else:
            st.error("No players detected. Try a different video.")
elif url is None:
    st.write("Waiting for video input... 📂")
