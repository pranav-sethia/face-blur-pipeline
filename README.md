# Face Blur

**Input:** a video file
**Output:** the same video, with faces blurred

## Build

```bash
docker build -t face-blur .
```

## Run

```bash
docker run -v /path/to/videos:/videos face-blur \
  --input /videos/my_video.mp4 \
  --output /videos/my_video_blurred.mp4
```

## Optional flags

- `--mode boxes` — draw boxes instead of blurring
- `--max-frames N` — only process the first N frames
