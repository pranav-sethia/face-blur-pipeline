# Validation evidence

Spot re-verification of the packaged pipeline, re-run end-to-end on 8 real-face clips and checked
against hand-labeled ground truth (recall) and a 10-frame buffer around each ground-truth window
(false positives — this is the same buffered-FP methodology the original validation used).

| Clip | Recall | False positives |
|---|---|---|
| logistics_unloading_boxes_into_truck_151 | 85.7% (96/112) | 0 |
| logistics_unloading_big_boxes_into_truck_152 | 100.0% (33/33) | 0 |
| logistics_unloading_big_boxes_from_truck_2_153 | 86.7% (13/15) | 0 |
| industrial_install_horn_capping_32 | 100.0% (7/7) | 0 |
| industrial_battery_housing_fitment_58 | 94.1% (16/17) | 0 |
| industrial_inserting_multiple_capacitor_PCB_107 | 93.4% (57/61) | 0 |
| logistics_keeping_clothes_folding_265 | 98.7% (77/78) | 0 |
| hospitality_kneading_formakingbread_dough_131 | 100.0% (5/5) | 0 |

**0 false positives on every clip checked**, recall 85.7–100%, consistent with the original
pipeline's own documented 91.5% recall / 0 FP across its 25-clip validation set.

The `*_annotated.mp4` files here are the green-box QA renders used to visually confirm tracking
behavior (`blur_faces.py --mode boxes`) — kept as evidence, not part of what the shipped tool needs
to run.
