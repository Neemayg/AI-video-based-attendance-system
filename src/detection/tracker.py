import numpy as np
import datetime
from typing import List, Dict, Optional
from src.detection.person_detector import PersonDetection

class TrackedPerson:
    def __init__(self, track_id: str, box: np.ndarray, last_seen: datetime.datetime):
        self.track_id = track_id
        self.box = box
        self.last_seen = last_seen

class CentroidTracker:
    def __init__(self, max_disappeared_seconds=3.0, max_distance=150.0):
        self.next_track_id = 1
        self.tracks: Dict[str, TrackedPerson] = {}  # track_id -> TrackedPerson
        self.max_disappeared_seconds = max_disappeared_seconds
        self.max_distance = max_distance

    def update(self, detections: List[PersonDetection], current_time: datetime.datetime):
        """
        detections: list of PersonDetection
        Returns: 
            assigned_track_ids: list of track IDs corresponding to the input detections
            active_tracks: dict mapping all currently active track_ids to their TrackedPerson objects
        """
        assigned_track_ids = [None] * len(detections)
        
        if len(detections) == 0:
            expired_tracks = []
            for track_id, track_data in self.tracks.items():
                time_missing = (current_time - track_data.last_seen).total_seconds()
                if time_missing > self.max_disappeared_seconds:
                    expired_tracks.append(track_id)
            
            for track_id in expired_tracks:
                del self.tracks[track_id]
                
            return assigned_track_ids, self.tracks.copy()

        # Calculate centroids for incoming rects
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_boxes = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.box
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            input_boxes.append(det.box)

        if len(self.tracks) == 0:
            for i in range(len(input_centroids)):
                tid_str = str(self.next_track_id)
                self.tracks[tid_str] = TrackedPerson(
                    track_id=tid_str,
                    box=input_boxes[i],
                    last_seen=current_time
                )
                assigned_track_ids[i] = tid_str
                self.next_track_id += 1
        else:
            track_ids = list(self.tracks.keys())
            
            # Reconstruct centroids from boxes for active tracks
            track_centroids = []
            for tid in track_ids:
                bx = self.tracks[tid].box
                tcx = int((bx[0] + bx[2]) / 2.0)
                tcy = int((bx[1] + bx[3]) / 2.0)
                track_centroids.append((tcx, tcy))
            track_centroids = np.array(track_centroids)

            # Calculate distance between existing tracks and new inputs
            D = np.linalg.norm(track_centroids[:, np.newaxis] - input_centroids, axis=2)

            if D.size > 0:
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_rows = set()
                used_cols = set()

                for row, col in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue

                    if D[row, col] > self.max_distance:
                        continue

                    track_id = track_ids[row]
                    self.tracks[track_id].box = input_boxes[col]
                    self.tracks[track_id].last_seen = current_time
                    assigned_track_ids[col] = track_id
                    
                    used_rows.add(row)
                    used_cols.add(col)

                # Check for expired tracks
                unused_rows = set(range(len(track_centroids))) - used_rows
                expired_tracks = []
                for row in unused_rows:
                    track_id = track_ids[row]
                    time_missing = (current_time - self.tracks[track_id].last_seen).total_seconds()
                    if time_missing > self.max_disappeared_seconds:
                        expired_tracks.append(track_id)
                
                for track_id in expired_tracks:
                    del self.tracks[track_id]

                # Check for new tracks
                unused_cols = set(range(len(input_centroids))) - used_cols
                for col in unused_cols:
                    tid_str = str(self.next_track_id)
                    self.tracks[tid_str] = TrackedPerson(
                        track_id=tid_str,
                        box=input_boxes[col],
                        last_seen=current_time
                    )
                    assigned_track_ids[col] = tid_str
                    self.next_track_id += 1

        return assigned_track_ids, self.tracks.copy()
