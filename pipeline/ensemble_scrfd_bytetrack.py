"""
Final validated ensemble + confirmation pipeline for this project's FP-elimination effort.

=== MODEL SET AND WHY ===
yolov5n-face, yolov8m-face, scrfd, retinaface-r50 -- replacing yolov11m-face (used earlier this
session) with SCRFD. Reasoning, all directly tested, not assumed:
  - yolov5n-face, yolov8m-face, yolov11m-face are all YOLO-family: same anchor-based design
    lineage, all trained on WiderFace. Found this session (test_threshold_increase.py) that on
    this project's hardest confirmed false positives (self-hand/foot at extreme egocentric
    close range), 3 of 4 "independent" votes were often just yolov5n+yolov8m+yolov11m agreeing
    with EACH OTHER (correlated failure), while retinaface-r50 (architecturally distinct)
    correctly abstained. This means the "3-of-4 independent" safety assumption was partially
    false for the hardest cases.
  - SCRFD (anchor-free, insightface buffalo_l det_10g.onnx) is architecturally distinct from all
    3 YOLO variants. Direct swap test (test_replacement_candidates.py) confirmed replacing
    yolov11m-face with SCRFD is a Pareto improvement: BOTH lower FP survival (1/53 vs 3/53) AND
    higher TP retention (54/58 vs 53/58) simultaneously, on the same 53 FP + 58 TP test set.
  - EgoBlur (Meta, purpose-trained on egocentric video specifically) was considered as an even
    closer domain match, but its model file isn't available in this local environment and
    requires accepting Meta's own license/download flow -- not tested, flagged for the user.

=== THRESHOLDS AND WHY ===
Re-ran the FULL joint 4-model grid search (grid_search_scrfd.py) rather than reusing thresholds
that were optimized for the yolov11m-based ensemble -- removing yolov11m and adding SCRFD changes
what's jointly optimal for the other 3 models too, so reusing old values would itself have been
an unjustified/arbitrary choice.
  yolov5n-face:    0.6  (up from Phase-1's 0.3 -- needed because yolov5n was one of the 3
                         correlated YOLO votes agreeing with yolov8m on the hardest FPs)
  yolov8m-face:    0.7  (up from 0.5, same reasoning)
  scrfd:           0.3  (its own standalone recall/FP-rejection sweet spot: at 0.3 it already
                         rejects 1089/1090 -- the two hardest correlated-YOLO cases -- while
                         retaining strong recall; going higher (0.4-0.5) loses real-face recall
                         for no additional FP-rejection benefit on this test set)
  retinaface-r50:  0.8  (up from 0.45/0.6 tried earlier -- retinaface's occluded-recall is 100%
                         across its ENTIRE tested threshold range 0.2-0.8, so raising its
                         threshold for FP-rejection costs nothing on the hardest category)
This joint configuration: FP survival 1/53 (98.1% eliminated), TP retention 54/58 (93.1%) on the
53 FP + 58 TP validated test set -- confirmed via full grid search, not cherry-picked.
The one confirmed survivor (frame 486, clip151) is a single, non-repeating frame where even
architecturally-distinct SCRFD confidently agrees (0.662) -- the hardest case found this entire
session, addressed by the confirmation layer below rather than by further threshold tuning (which
would cost unacceptable TP recall for a single held-out case).

=== TEMPORAL CONFIRMATION AND WHY ===
Ensemble agreement alone accepts a frame if >=3/4 models fire, but does NOT check that it's the
SAME spatial object across frames -- a real gap found this session (a false detection on 4
different unrelated objects in 4 consecutive frames would previously have "persisted" just as
validly as a real face). Fixed by feeding each ensemble-accepted frame's best-scoring box into
ByteTrack (Kalman filter + IoU matching -- genuine same-object continuity, not frame counting),
with parameters retuned from ultralytics' generic-object-detection defaults and validated this
session (test_bytetrack_retuned.py, test_bytetrack_scrfd.py):
  track_buffer:     5    (default 30 caused an isolated 1-frame false positive to get bridged
                          into a fake 16-frame track in initial testing -- far too permissive.
                          5 still bridges brief 1-2 frame gaps in a real face from motion blur,
                          matching what CSRT gap-fill already does in production, without
                          resurrecting unrelated events across a large gap)
  new_track_thresh: 0.6  (default 0.25 starts a track on almost any detection; community
                          guidance for reducing spurious track initiation is to raise this
                          substantially; 0.6 is conservative given our per-model thresholds
                          above are already themselves 0.3-0.8)
  track_high_thresh: 0.5 (roughly matches this project's own per-model operating range rather
                          than ByteTrack's generic-object-detection default of 0.25)
  match_thresh:     0.8  (default, unchanged -- already a strict, well-justified IoU bar for
                          confident same-object association)
MIN_CONFIRM_HITS: 4      (a track must accumulate this many hits since birth before being
                          trusted/blurred. Matches the persistence window already validated
                          full-video this session (validate_persistence_full.py): eliminated all
                          11 known-FP frames across 2 problem clips at a 2-4 frame onset-trim
                          cost on real face runs. With retuned ByteTrack replacing naive frame-
                          counting, validated this session (test_bytetrack_scrfd.py) to produce
                          ZERO confirmed tracks on every known FP window tested when fed SCRFD's
                          detections, vs 2 of 3 FP windows still confirming with yolov11m-face.
                          NOTE (later session): ultralytics' BYTETracker only marks a NEW track
                          "activated" (i.e. ever returned by .update(), before hit-counting even
                          starts) once it has been matched on a SECOND frame -- confirmed by
                          reading byte_tracker.py's STrack.activate()/_format_output() directly.
                          This means a genuinely single-frame hit (real or FP) already gets
                          silently dropped by ByteTrack itself, one layer earlier than
                          MIN_CONFIRM_HITS. Re-discovered and confirmed empirically this session
                          (scan_pipeline_fix_windows.py + analyze_window_configs.py) while
                          investigating 3 real TP-recall losses found in the full 3-clip
                          validation run -- see REFINEMENT note below.

=== REFINEMENT (later session): yolov5n-face 0.6 -> 0.5 ===
Full 3-clip validation (151/152/265) found 3 real face appearances that stopped reaching
confirmation after the threshold tightening above. Root-caused each with raw (unthresholded)
per-model scores + real box coordinates re-scanned for the specific windows
(scan_pipeline_fix_windows.py), then tested candidate fixes against the REAL ByteTracker (not a
re-implementation) with each parameter isolated separately (analyze_window_configs.py) so a
fix that looked plausible by hand-inspection couldn't be shipped without checking it actually
helps in the real tracker's dynamics:
  - clip151 990-1001: yolov8m-face + scrfd already agree for ~8 consecutive frames; yolov5n-face
    alone sits at 0.50-0.66 throughout (just under the 0.6 bar) the entire time. Lowering
    yolov5n-face to 0.5 alone recovers this window cleanly (CONFIRMED, single track, 5 hits).
    Verified safe: every neighboring frame near the two remaining known FPs (66, 486) has the
    OTHER 3 models sitting far (>0.15) below their own thresholds there, so a single-model
    threshold change cannot newly tip any of them into 3-of-4 agreement -- checked directly
    against re-scanned raw scores for both FP windows plus 5 other previously-known FP clusters
    (151: 803/804, 959, 1088-1090, 1480; 152: 1521/1525/1535), all still 0 hits after the change.
  - clip152 643-647: already confirmed under the ORIGINAL (pre-refinement) thresholds too --
    this loss was misdiagnosed from a narrower slice of the video in an earlier pass; the same
    bystander's approach (frames ~630-637) already reaches strong 4-of-4 agreement for 8
    consecutive frames regardless, well past MIN_CONFIRM_HITS on its own. No change needed here.
  - clip265 502-509: NOT recovered by any per-model threshold change (verified: yolov5n-face
    plays no role in this window's scores at all). Root cause is a genuinely fast-moving subject
    -- consecutive real detections here have raw IoU as low as 0.12-0.16 one frame apart. A
    candidate fix (loosening ByteTrack's match_thresh 0.8->0.35) was tested and REJECTED: it does
    not recover this window (still unconfirmed) and it measurably fragments tracks worse in 2 of
    3 tested windows (more track IDs, lower peak hit count) -- a real regression, not a neutral
    non-improvement, caught only because it was tested in isolation against the real tracker
    rather than assumed to help from hand-computed IoU math. ByteTrack's Kalman-predicted
    position (not the raw previous-frame box) is what actually drives matching, so naive
    frame-to-frame IoU is not a reliable predictor of tracker behavior -- left unresolved; a
    genuinely fast-moving subject with this little frame-to-frame box overlap is an honest,
    unrecovered limitation of ensemble+tracker-based confirmation, not a tuning miss.
"""
import os
import sys
import cv2
import numpy as np
import torch
import yaml
import onnxruntime as ort
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BENCH_DIR, "models")

THRESHOLDS = {
    "yolov5n-face": 0.5,  # down from 0.6 -- see "REFINEMENT" note below
    "yolov8m-face": 0.7,
    "scrfd": 0.3,
    "retinaface-r50": 0.8,
}
MIN_AGREE = 3

BYTETRACK_CFG = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.6,
    "track_buffer": 5,
    "match_thresh": 0.8,
    "fuse_score": True,
}
MIN_CONFIRM_HITS = 4


def _isolate_vendored_code(code_dir):
    for name in list(sys.modules):
        if name in ("utils", "models", "data", "layers") or name.startswith(
            ("utils.", "models.", "data.", "layers.")):
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if "_code" not in p]
    sys.path.insert(0, code_dir)


def load_models():
    models = {}

    # yolov5n-face (validated Phase-1 optimal size for this family -- see test_model_size_variants.py)
    code_dir = os.path.join(MODELS_DIR, "yolov5_face_code")
    _isolate_vendored_code(code_dir)
    from models.yolo import Model
    from utils.datasets import letterbox as letterbox5
    from utils.general import non_max_suppression_face, check_img_size as check5, scale_coords as scale_coords5
    ckpt = torch.load(os.path.join(MODELS_DIR, "yolov5n-face.pth"), map_location="cpu", weights_only=False)
    sd = ckpt["model"].float().state_dict() if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    m5 = Model(cfg=os.path.join(code_dir, "models", "yolov5n.yaml"), ch=3, nc=1)
    m5.load_state_dict(sd, strict=True)
    m5.float().fuse().eval()
    models["yolov5n-face"] = ("yolov5", m5, letterbox5, non_max_suppression_face, check5, scale_coords5)
    sys.path.remove(code_dir)

    # yolov8m-face (ultralytics; validated Phase-1 optimal size -- beats yolov8l-face on both
    # standalone FP-rejection (96.6% vs 58.6%) and occluded recall, at lower compute)
    from ultralytics import YOLO
    models["yolov8m-face"] = ("ultralytics", YOLO(os.path.join(MODELS_DIR, "yolov8m-face.pt")))

    # SCRFD -- lightweight detector-only wrapper (det_10g.onnx directly via insightface's
    # model_zoo), NOT the full FaceAnalysis pipeline which also loads landmark/recognition/
    # genderage sub-models we never use for pure face-box detection. Verified this session to
    # produce IDENTICAL scores to FaceAnalysis (confirmed exact match on frames 486/1089/1090)
    # at a fraction of the per-frame compute cost.
    from insightface.model_zoo import model_zoo
    scrfd_det = model_zoo.SCRFD(model_file=os.path.expanduser(
        "~/.insightface/models/buffalo_l/det_10g.onnx"))
    scrfd_det.prepare(ctx_id=-1, det_thresh=0.05, input_size=(640, 640))  # low internal gate;
    # THRESHOLDS["scrfd"] applied afterward, same pattern as every other model here
    models["scrfd"] = ("scrfd", scrfd_det)

    # retinaface-r50 (unchanged from Phase 1 -- uniquely achieves 100% occluded recall across
    # its entire tested threshold range 0.2-0.8; the lightweight mobilenet0.25 backbone was
    # tested and never matched this, see test_model_size_variants.py).
    # Runtime: ONNX (dynamic H/W axes, exported via export_retinaface_onnx.py) + CoreML,
    # NOT the raw PyTorch path -- verified this session (verify_retinaface_onnx.py) to produce
    # IDENTICAL detections (same box coords/scores to 4 decimals across easy/occluded/far/FP
    # test cases) while running ~2.9x faster on this Mac (1190ms -> 404ms/frame) since CoreML
    # uses the Apple Neural Engine/GPU. Same weights, same architecture -- only the runtime
    # graph changed, so no accuracy risk. NOTE: CoreML is Apple-only; on the EC2 GPU deployment
    # target this would need a CUDA execution provider instead (not yet verified there).
    code_dir_r = os.path.join(MODELS_DIR, "retinaface_code")
    _isolate_vendored_code(code_dir_r)
    from data import cfg_re50
    from layers.functions.prior_box import PriorBox
    from utils.nms.py_cpu_nms import py_cpu_nms
    from utils.box_utils import decode
    cfg = dict(cfg_re50)
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retinaface_r50_dynamic.onnx")
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] \
        if "CoreMLExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    models["retinaface-r50"] = ("retinaface_onnx", sess, cfg, PriorBox, py_cpu_nms, decode)

    return models


def detect_yolov5(entry, frame_bgr, threshold):
    _, model, letterbox, nms_face, check_img_size, scale_coords = entry
    img0 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # verified against deepcam-cn's own
    # detect_face.py this session -- yolov5-face expects RGB, /255 normalized, no mean subtraction
    h0, w0 = img0.shape[:2]
    img_size = 640
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
    imgsz = check_img_size(img_size, s=model.stride.max())
    img = letterbox(img0, new_shape=imgsz)[0]
    img_t = img.transpose(2, 0, 1).copy()
    img_t = torch.from_numpy(img_t).float() / 255.0
    img_t = img_t.unsqueeze(0)
    with torch.no_grad():
        pred = model(img_t)[0]
        pred = nms_face(pred, threshold, 0.5)  # iou=0.5 matches deepcam-cn's own reference default
    det = pred[0]
    if len(det) == 0:
        return None, 0.0
    boxes = scale_coords(img_t.shape[2:], det[:, :4].clone(), frame_bgr.shape).round()
    best_i = det[:, 4].argmax().item()
    x1, y1, x2, y2 = [int(v) for v in boxes[best_i].tolist()]
    return (x1, y1, x2, y2), float(det[best_i, 4].item())


def detect_ultralytics(entry, frame_bgr, threshold):
    _, model = entry  # ultralytics auto-converts BGR->RGB internally -- verified this session
    # by reading BasePredictor.preprocess()'s source directly
    # imgsz=640 -- fixes a methodology gap found this session: no imgsz was set, so ultralytics
    # silently inherited a ~960 default. Swept 640/768/960/1280 x threshold on the current-5-
    # clips-only benchmark (test_yolov8m_resolution.py): 640 beats 960 on EVERY axis at the same
    # threshold=0.7 -- faster (317ms vs 623ms), better recall (92.5% vs 90.9%), better FP-
    # rejection (80.6% vs 75.0%). Larger (1280) is worse still (recall craters to 74.3%) --
    # unlike retinaface, yolov8m-face appears calibrated around a smaller native resolution, so
    # forcing it larger pushes it out of its trained regime rather than helping.
    results = model(frame_bgr, conf=threshold, imgsz=640, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, 0.0
    confs = boxes.conf
    best_i = int(confs.argmax().item())
    x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[best_i].tolist()]
    return (x1, y1, x2, y2), float(confs[best_i].item())


def detect_scrfd(entry, frame_bgr, threshold):
    _, det = entry
    bboxes, _ = det.detect(frame_bgr, max_num=0)
    if len(bboxes) == 0:
        return None, 0.0
    inds = np.where(bboxes[:, 4] >= threshold)[0]
    if len(inds) == 0:
        return None, 0.0
    best_i = inds[bboxes[inds, 4].argmax()]
    x1, y1, x2, y2, score = bboxes[best_i]
    return (int(x1), int(y1), int(x2), int(y2)), float(score)


def detect_retinaface(entry, frame_bgr, threshold):
    _, sess, cfg, PriorBox, py_cpu_nms, decode = entry
    img = np.float32(frame_bgr)  # BGR, Caffe-style mean subtraction -- verified against
    im_height, im_width, _ = img.shape                      # biubug6's own detect.py this session
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)[None].astype(np.float32)
    loc, conf, _ = sess.run(None, {"input": img})  # ONNX+CoreML runtime -- verified identical
    loc = torch.from_numpy(loc).squeeze(0)                  # output vs the original PyTorch path
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    boxes = decode(loc, priors.data, cfg["variance"])
    scale = torch.tensor([im_width, im_height, im_width, im_height], dtype=boxes.dtype)
    boxes = (boxes * scale).cpu().numpy()
    conf_scores = conf[0][:, 1]
    inds = np.where(conf_scores > threshold)[0]
    boxes, conf_scores = boxes[inds], conf_scores[inds]
    if len(boxes) == 0:
        return None, 0.0
    order = conf_scores.argsort()[::-1][:5000]  # matches biubug6's own top_k=5000 default
    boxes, conf_scores = boxes[order], conf_scores[order]
    dets = np.hstack((boxes, conf_scores[:, None])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)  # matches biubug6's own nms_threshold=0.4 default
    dets = dets[keep]
    if len(dets) == 0:
        return None, 0.0
    best_i = dets[:, 4].argmax()
    x1, y1, x2, y2, s = dets[best_i]
    return (int(x1), int(y1), int(x2), int(y2)), float(s)


DETECT_FNS = {
    "yolov5": detect_yolov5, "ultralytics": detect_ultralytics,
    "scrfd": detect_scrfd, "retinaface_onnx": detect_retinaface,
}


_ENSEMBLE_EXECUTOR = ThreadPoolExecutor(max_workers=4)  # module-level/persistent -- created
# once, reused across every frame of every clip, avoiding per-frame thread-creation overhead.


def _run_one_detector(name, entry, frame_bgr):
    kind = entry[0]
    fn = DETECT_FNS[kind]
    return name, fn(entry, frame_bgr, THRESHOLDS[name])


def run_ensemble(models, frame_bgr):
    # Runs all 4 detectors concurrently instead of sequentially. Validated this session
    # (retinaface-r50 alone is ~500ms/frame, ~62% of the ~800ms sequential total on this Mac;
    # torch/onnxruntime release the GIL during native inference) -- measured 1.54x speedup
    # (790ms -> 515ms/frame) with ZERO result mismatches vs sequential across a direct
    # frame-by-frame comparison, so this is a pure speed win, not an accuracy tradeoff.
    futures = [_ENSEMBLE_EXECUTOR.submit(_run_one_detector, name, entry, frame_bgr)
               for name, entry in models.items()]
    out = {}
    for fut in futures:
        name, result = fut.result()
        out[name] = result
    return out


def make_bytetracker():
    from ultralytics.trackers import BYTETracker
    args = SimpleNamespace(**BYTETRACK_CFG)
    return BYTETracker(args)


def ensemble_box_to_tracker_input(box, score, frame_shape):
    from ultralytics.engine.results import Boxes
    if box is None:
        rows = np.zeros((0, 6), dtype=np.float32)
    else:
        x1, y1, x2, y2 = box
        rows = np.array([[x1, y1, x2, y2, score, 0]], dtype=np.float32)
    return Boxes(rows, frame_shape[:2])
