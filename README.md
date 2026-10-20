# Face Blur

Blurs faces in a video. 4-model detector ensemble (SCRFD, YOLOv5n-face, YOLOv8m-face,
RetinaFace-R50) + ByteTrack, tuned for 0 false positives on egocentric/head-mounted footage.

**Input:** one video file (any format OpenCV can read — .mp4, .mov, etc.)
**Output:** the same video, same length/resolution, with every confirmed face Gaussian-blurred.

## 1. Build the image (one-time setup)

```bash
docker build -t face-blur .
```

## 2. Run it on a video

Mount the folder your video is in, point `--input`/`--output` at files inside that mount, get the
blurred video back out at `--output`.

```bash
docker run -v /path/to/videos:/videos face-blur \
  --input /videos/my_video.mp4 \
  --output /videos/my_video_blurred.mp4
```

That's it — no Python setup, no model downloads, nothing else to install.

### Optional flags
- `--mode boxes` — draws a green box instead of blurring, to visually check what's being tracked.
- `--max-frames N` — only process the first N frames, for a quick test before running a full video.

### Speed
CPU-only, no GPU required — but not fast: roughly 15–20 minutes per ~2000 frames. Plan accordingly
for long videos.

## Validated performance

91.5% recall, 0 false positives on the original 25-clip internal validation set. Re-checked on 8
more real-face clips before packaging — 0 false positives on all 8, recall 85.7–100%. Details and
diagnostic CSVs in `qa_evidence/VALIDATION.md`.

## One honest note

The detection/tracking algorithm (`pipeline/ensemble_scrfd_bytetrack.py`) is the original,
unchanged model/threshold/tracking configuration. The driver that runs it on a video
(`pipeline/blur_faces.py`) was rebuilt and reverified against the original's own archived ground
truth, since the original driver script was lost in an earlier cleanup.
