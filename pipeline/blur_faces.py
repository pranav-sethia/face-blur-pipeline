"""
Face-blur CLI: runs the validated 4-model ensemble + ByteTrack confirmation pipeline
(ensemble_scrfd_bytetrack.py, unchanged) on a video and blurs every confirmed face.

Two-pass design: Pass 1 runs detection + tracking + confirmation and records, for every frame,
each track's box and whether that track ever gets confirmed by the end of the clip. A track is
only trusted once it accumulates MIN_CONFIRM_HITS real hits -- but once it does, it's retroactively
treated as confirmed all the way back to its own first frame, not just forward from the hit that
crossed the threshold. That retroactive decision can't be applied in a single streaming pass, since
earlier frames would already be written to the output file before the confirming frame is even
read. Pass 2 re-reads the video and renders every frame using that final, complete answer.

Box position during a tracked-but-not-currently-detected gap (ByteTrack bridging a brief miss):
linearly interpolates between the real detected box right before the gap and the real detected box
right after the same track resumes (ByteTrack keeps the same track_id across a bridged gap). Falls
back to ByteTrack's own Kalman-predicted position only if the track never resumes.

--mode blur (default) applies this repo's documented blur spec: expand each box 1.3x centered on
the detection (clamped to frame), Gaussian kernel max(99, int(box_width * 0.6)) forced odd.
--mode boxes draws a green box instead of blurring, for visually checking tracking behavior.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from ensemble_scrfd_bytetrack import (
    load_models,
    run_ensemble,
    make_bytetracker,
    ensemble_box_to_tracker_input,
    MIN_AGREE,
    MIN_CONFIRM_HITS,
)

BOX_COLOUR = (0, 255, 0)  # green, --mode boxes only


def _select_ensemble_box(detector_results):
    agreeing = [(name, box, score) for name, (box, score) in detector_results.items() if box is not None]
    n_agree = len(agreeing)
    if n_agree < MIN_AGREE:
        return None, 0.0, n_agree
    agreeing.sort(key=lambda t: t[2], reverse=True)
    _, best_box, best_score = agreeing[0]
    return best_box, best_score, n_agree


def _expand_bbox(x1, y1, x2, y2, frame_w, frame_h, scale=1.3):
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    nx1 = int(max(0, cx - w / 2))
    ny1 = int(max(0, cy - h / 2))
    nx2 = int(min(frame_w - 1, cx + w / 2))
    ny2 = int(min(frame_h - 1, cy + h / 2))
    return nx1, ny1, nx2, ny2


def _compute_kernel(face_width):
    k = max(99, int(face_width * 0.6))
    return k + 1 if k % 2 == 0 else k


def _blur_box(frame, box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    k = _compute_kernel(x2 - x1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)


def process_video(models, input_path, output_path, mode="blur", max_frames=None,
                   dump_diagnostic_csv=None):
    """Runs the two-pass pipeline on input_path and writes the result to output_path.
    Returns (n_frames, n_confirmed_frames)."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    # --- Pass 1: detection + tracking + confirmation, no video writing ---
    tracker = make_bytetracker()
    backfilled_track_ids = set()
    hit_count = {}  # our own cumulative hit count, not ByteTrack's internal tracklet_len, which
                     # resets to 0 every time a track is briefly lost then re-found -- counting our
                     # own persistent total matches the pipeline's documented "hits since birth" rule
    track_presence = []  # [frame_idx] -> list of (track_id, pool)  ("tracked" or "lost")
    diag_rows = []
    real_box = {}    # (track_id, frame_idx) -> box, only for real ("tracked") matches
    kalman_box = {}  # (track_id, frame_idx) -> box, ByteTrack's own predicted position, every frame

    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        detector_results = run_ensemble(models, frame)
        box, score, n_agree = _select_ensemble_box(detector_results)
        tracker_input = ensemble_box_to_tracker_input(box, score, frame.shape)
        tracker.update(tracker_input, frame)

        for t in tracker.tracked_stracks:
            hit_count[t.track_id] = hit_count.get(t.track_id, 0) + 1
            if hit_count[t.track_id] >= MIN_CONFIRM_HITS:
                backfilled_track_ids.add(t.track_id)

        frame_entries = []
        for t in tracker.tracked_stracks:
            x1, y1, x2, y2 = (int(v) for v in t.xyxy.tolist())
            real_box[(t.track_id, frame_idx)] = (x1, y1, x2, y2)
            kalman_box[(t.track_id, frame_idx)] = (x1, y1, x2, y2)
            frame_entries.append((t.track_id, "tracked"))
        for t in tracker.lost_stracks:
            kalman_box[(t.track_id, frame_idx)] = tuple(int(v) for v in t.xyxy.tolist())
            frame_entries.append((t.track_id, "lost"))
        track_presence.append(frame_entries)
        frame_idx += 1

    cap.release()
    n_frames_seen = frame_idx

    # Resolve a box for every (track_id, frame_idx) a track is present at, including "lost"
    # frames: interpolate between the real box before/after a bridged gap when the same track_id
    # resumes, otherwise fall back to ByteTrack's own live Kalman prediction for that frame.
    real_frames_by_track = {}
    for (tid, f) in real_box:
        real_frames_by_track.setdefault(tid, []).append(f)
    for tid in real_frames_by_track:
        real_frames_by_track[tid].sort()

    def resolve_box(track_id, fidx):
        key = (track_id, fidx)
        if key in real_box:
            return real_box[key]
        frames = real_frames_by_track.get(track_id, [])
        before = max((f for f in frames if f < fidx), default=None)
        after = min((f for f in frames if f > fidx), default=None)
        if before is not None and after is not None:
            b1, b2 = real_box[(track_id, before)], real_box[(track_id, after)]
            t = (fidx - before) / (after - before)
            return tuple(int(round(v1 + (v2 - v1) * t)) for v1, v2 in zip(b1, b2))
        return kalman_box[key]

    boxes_per_frame = [
        [(tid, *resolve_box(tid, fidx)) for tid, pool in entries]
        for fidx, entries in enumerate(track_presence)
    ]

    # --- Pass 2: render, using the FINAL confirmed-track-id set ---
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n_confirmed_frames = 0
    diag_rows = []
    for frame_idx in range(n_frames_seen):
        ret, frame = cap.read()
        if not ret:
            break

        confirmed_this_frame = False
        for (track_id, x1, y1, x2, y2) in boxes_per_frame[frame_idx]:
            confirmed = track_id in backfilled_track_ids
            if dump_diagnostic_csv is not None:
                diag_rows.append((frame_idx, track_id, confirmed, x1, y1, x2, y2))
            if not confirmed:
                continue
            confirmed_this_frame = True
            if mode == "blur":
                _blur_box(frame, _expand_bbox(x1, y1, x2, y2, w, h))
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOUR, 2)

        if confirmed_this_frame:
            n_confirmed_frames += 1
        writer.write(frame)

    cap.release()
    writer.release()

    if dump_diagnostic_csv is not None:
        with open(dump_diagnostic_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["frame_idx", "track_id", "confirmed", "x1", "y1", "x2", "y2"])
            wcsv.writerows(diag_rows)

    return n_frames_seen, n_confirmed_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source video")
    parser.add_argument("--output", required=True, help="Path to write the result video to")
    parser.add_argument("--mode", choices=["blur", "boxes"], default="blur",
                         help="blur (default): blur every confirmed face. "
                              "boxes: draw a green box instead, for checking tracking visually.")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Only process the first N frames (quick test).")
    parser.add_argument("--dump-diagnostic-csv", default=None,
                         help="Optional path to write a per-frame/per-track diagnostic CSV.")
    args = parser.parse_args()

    print("Loading models...", flush=True)
    models = load_models()
    print(f"Processing {args.input} (mode={args.mode})...", flush=True)
    n_frames, n_confirmed = process_video(
        models, args.input, args.output, mode=args.mode,
        max_frames=args.max_frames, dump_diagnostic_csv=args.dump_diagnostic_csv,
    )
    verb = "blurred" if args.mode == "blur" else "boxed"
    print(f"Done: wrote {args.output} ({n_frames} frames, {n_confirmed} {verb} frames)", flush=True)


if __name__ == "__main__":
    main()
