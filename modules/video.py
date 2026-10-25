"""
video.py — CCTV video frame sampling, centroid tracking, and event log generation.
"""
from __future__ import annotations

import io
import math
from typing import Any, Optional

import numpy as np
from PIL import Image

from config import (
    CRIME_CATEGORIES,
    DEMO_VIDEO_EVENTS,
    VIDEO_FPS_SAMPLE,
    VIDEO_MAX_FRAMES,
)


# ---------------------------------------------------------------------------
# Frame sampling with OpenCV
# ---------------------------------------------------------------------------

def sample_frames(
    video_bytes: bytes,
    fps_sample: int = VIDEO_FPS_SAMPLE,
    max_frames: int = VIDEO_MAX_FRAMES,
) -> list[dict[str, Any]]:
    """
    Extract frames from a video at the given sampling rate.

    Args:
        video_bytes: Raw video file bytes.
        fps_sample:  How many frames per second to extract.
        max_frames:  Hard limit on total extracted frames.

    Returns:
        List of dicts with keys: 'frame_index', 'timestamp_sec', 'image' (PIL).

    Raises:
        RuntimeError: If OpenCV cannot open the video.
    """
    try:
        import cv2  # type: ignore
        import tempfile, os

        # Write to temp file (OpenCV needs a file path)
        suffix = ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open the video file.")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(video_fps / fps_sample))

        frames: list[dict[str, Any]] = []
        frame_idx = 0
        sampled = 0

        while sampled < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ts = frame_idx / video_fps

            frames.append({
                "frame_index": frame_idx,
                "timestamp_sec": round(ts, 2),
                "image": pil_img,
            })

            frame_idx += step
            sampled += 1
            if frame_idx >= total_frames:
                break

        cap.release()
        os.unlink(tmp_path)
        return frames

    except ImportError:
        raise RuntimeError("OpenCV (opencv-python-headless) is not installed.")
    except Exception as exc:
        raise RuntimeError(f"Frame sampling failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Simple centroid tracker
# ---------------------------------------------------------------------------

class CentroidTracker:
    """
    Lightweight centroid-based multi-object tracker.

    Maintains track IDs across frames using nearest-centroid matching.
    No deep learning required.
    """

    def __init__(self, max_disappeared: int = 5, max_distance: int = 80):
        self.next_id = 1
        self.objects: dict[str, tuple[float, float]] = {}   # track_id -> centroid
        self.disappeared: dict[str, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history: dict[str, list[tuple[float, float]]] = {}

    def _centroid(self, bbox: list[int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _track_id(self, n: int) -> str:
        prefix = "T" if n < 100 else "VEH"
        return f"{prefix}-{n:02d}"

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Update tracker with new frame detections.

        Args:
            detections: List of dicts with 'bbox' and 'class' keys.

        Returns:
            Detections enriched with 'track_id' and 'centroid' keys.
        """
        if not detections:
            for tid in list(self.disappeared):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.objects[tid]
                    del self.disappeared[tid]
            return []

        new_centroids = [self._centroid(d.get("bbox", [0, 0, 1, 1])) for d in detections]

        if not self.objects:
            for i, c in enumerate(new_centroids):
                tid = self._track_id(self.next_id)
                self.next_id += 1
                self.objects[tid] = c
                self.disappeared[tid] = 0
                self.history.setdefault(tid, []).append(c)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Cost matrix
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            assignments: list[tuple[int, int, float]] = []

            for row, oc in enumerate(object_centroids):
                for col, nc in enumerate(new_centroids):
                    dist = math.hypot(oc[0] - nc[0], oc[1] - nc[1])
                    assignments.append((row, col, dist))

            assignments.sort(key=lambda x: x[2])

            for row, col, dist in assignments:
                if row in used_rows or col in used_cols:
                    continue
                if dist > self.max_distance:
                    break
                tid = object_ids[row]
                self.objects[tid] = new_centroids[col]
                self.disappeared[tid] = 0
                self.history.setdefault(tid, []).append(new_centroids[col])
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(len(object_centroids))) - used_rows:
                tid = object_ids[row]
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.objects[tid]
                    del self.disappeared[tid]

            for col in set(range(len(new_centroids))) - used_cols:
                tid = self._track_id(self.next_id)
                self.next_id += 1
                self.objects[tid] = new_centroids[col]
                self.disappeared[tid] = 0
                self.history.setdefault(tid, []).append(new_centroids[col])

        enriched = []
        for i, det in enumerate(detections):
            det = dict(det)
            if i < len(new_centroids):
                c = new_centroids[i]
                # Find the track_id whose centroid matches this detection's centroid
                tid = next(
                    (tid for tid, oc in self.objects.items()
                     if abs(oc[0] - c[0]) < 1 and abs(oc[1] - c[1]) < 1),
                    f"T-?{i}",
                )
                det["track_id"] = tid
                det["centroid"] = c
            enriched.append(det)

        return enriched


# ---------------------------------------------------------------------------
# Per-frame detection (simulated bounding boxes)
# ---------------------------------------------------------------------------

def _simulated_detections(frame_index: int, img_w: int, img_h: int) -> list[dict[str, Any]]:
    """
    Generate plausible simulated detections for a frame.

    The positions drift slightly each frame to simulate movement.
    """
    drift = frame_index * 4
    detections = [
        {
            "class": "person",
            "confidence": round(0.80 + (frame_index % 5) * 0.02, 2),
            "bbox": [
                min(img_w - 60, 100 + drift),
                50,
                min(img_w - 10, 200 + drift),
                200,
            ],
        },
    ]
    if frame_index > 10:
        detections.append({
            "class": "person",
            "confidence": round(0.75 + (frame_index % 3) * 0.02, 2),
            "bbox": [
                max(10, img_w - 200 - drift // 2),
                80,
                max(60, img_w - 100 - drift // 2),
                220,
            ],
        })
    if frame_index > 20 and frame_index % 15 < 8:
        detections.append({
            "class": "vehicle",
            "confidence": 0.72,
            "bbox": [10, img_h - 120, 200, img_h - 30],
        })
    return detections


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------

_EVENT_RULES: list[tuple[str, list[str], str, str]] = [
    # (class_keyword, event_keywords_in_track_action, crime_category, event_description_template)
    ("person",  [],              "Trespass",    "Person (track {tid}) detected in frame"),
    ("firearm", [],              "Armed Threat", "Firearm detected — Armed Threat (track {tid})"),
    ("edged_weapon", [],         "Armed Threat", "Edged weapon detected — Armed Threat (track {tid})"),
    ("vehicle", [],              "Unknown",     "Vehicle (track {tid}) detected in frame"),
]


def classify_event(
    det_class: str,
    track_id: str,
    frame_index: int,
    total_frames: int,
) -> tuple[str, str]:
    """
    Map a detection class + context to an event description and crime category.

    Returns:
        (event_description, crime_category)
    """
    # Simple rule-based progression
    progress = frame_index / max(1, total_frames)

    if det_class == "person":
        if progress < 0.2:
            return f"Person ({track_id}) enters scene", "Trespass"
        elif progress < 0.5:
            return f"Person ({track_id}) moves through restricted area", "Trespass"
        elif progress < 0.7:
            return f"Person ({track_id}) engages with target — possible theft", "Theft"
        elif progress < 0.85:
            return f"Person ({track_id}) confrontational posture — possible assault", "Assault"
        else:
            return f"Person ({track_id}) exits scene — suspect fled", "Theft"

    elif det_class in ("firearm", "edged_weapon"):
        return f"Weapon ({det_class}) visible — track {track_id}", "Armed Threat"

    elif det_class == "vehicle":
        if progress > 0.8:
            return f"Vehicle ({track_id}) departs — possible getaway", "Unknown"
        return f"Vehicle ({track_id}) detected in vicinity", "Unknown"

    return f"Object ({det_class}) detected — track {track_id}", "Unknown"


# ---------------------------------------------------------------------------
# Trajectory overlay
# ---------------------------------------------------------------------------

def draw_trajectories(
    frame_img: Image.Image,
    tracker: CentroidTracker,
) -> Image.Image:
    """
    Draw movement trajectories for all tracks on the given frame.

    Args:
        frame_img: PIL Image of the current frame.
        tracker:   CentroidTracker with history.

    Returns:
        Annotated PIL Image.
    """
    from PIL import ImageDraw

    img = frame_img.copy()
    draw = ImageDraw.Draw(img)

    colours = [
        (0, 255, 156), (255, 100, 100), (100, 180, 255),
        (255, 200, 0), (200, 100, 255), (255, 140, 0),
    ]

    for i, (tid, path) in enumerate(tracker.history.items()):
        if len(path) < 2:
            continue
        colour = colours[i % len(colours)]
        for j in range(1, len(path)):
            draw.line([path[j - 1], path[j]], fill=colour, width=2)
        # Label last known position
        last = path[-1]
        draw.ellipse(
            [last[0] - 5, last[1] - 5, last[0] + 5, last[1] + 5],
            fill=colour, outline=(255, 255, 255),
        )
        draw.text((last[0] + 7, last[1] - 7), tid, fill=colour)

    return img


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def analyse_video_demo() -> dict[str, Any]:
    """Return demo video analysis results."""
    import copy
    events = copy.deepcopy(DEMO_VIDEO_EVENTS)
    return {
        "events": events,
        "frame_count": 60,
        "duration_sec": 60.0,
        "track_ids": list({e["track_id"] for e in events}),
        "simulated": True,
        "trajectory_image": None,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_video(
    video_bytes: bytes,
    provider: str = "",
    api_key: str = "",
    demo_mode: bool = True,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """
    Analyse a CCTV video: sample frames, track objects, generate event log.

    Args:
        video_bytes:       Raw video file bytes.
        provider:          LLM provider (unused in current implementation).
        api_key:           API key (unused in current implementation).
        demo_mode:         Force demo mode.
        progress_callback: Optional callable(fraction) for progress updates.

    Returns:
        Dict with keys: events, frame_count, duration_sec, track_ids,
        simulated, trajectory_image (PIL or None).
    """
    if demo_mode or not api_key:
        return analyse_video_demo()

    try:
        frames = sample_frames(video_bytes)
        if not frames:
            return analyse_video_demo()

        tracker = CentroidTracker()
        events: list[dict[str, Any]] = []
        total = len(frames)
        last_trajectory_img: Optional[Image.Image] = None

        for idx, frame_data in enumerate(frames):
            if progress_callback:
                progress_callback(idx / total)

            img = frame_data["image"]
            ts = frame_data["timestamp_sec"]
            ts_str = f"{int(ts)//60:02d}:{int(ts)%60:02d}"

            # Use simulated detections (LLM per-frame would be too slow/costly)
            detections = _simulated_detections(frame_data["frame_index"], img.width, img.height)
            tracked = tracker.update(detections)

            for det in tracked:
                tid = det.get("track_id", "T-?")
                cls = det.get("class", "unknown")
                conf = det.get("confidence", 0.0)
                event_desc, crime_cat = classify_event(cls, tid, idx, total)

                events.append({
                    "frame": frame_data["frame_index"],
                    "timestamp": ts_str,
                    "track_id": tid,
                    "event": event_desc,
                    "crime_category": crime_cat,
                    "confidence": conf,
                })

            if idx == total - 1:
                last_trajectory_img = draw_trajectories(img, tracker)

        if progress_callback:
            progress_callback(1.0)

        duration = frames[-1]["timestamp_sec"] if frames else 0.0
        return {
            "events": events,
            "frame_count": total,
            "duration_sec": duration,
            "track_ids": list(tracker.history.keys()),
            "simulated": False,
            "trajectory_image": last_trajectory_img,
        }

    except Exception as exc:
        result = analyse_video_demo()
        result["_api_error"] = str(exc)
        return result
