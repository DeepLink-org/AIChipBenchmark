import os
from pathlib import Path

CASES = [
    'bbox2delta', 'bbox_overlaps', 'delta2bbox', 'l2_loss', 'batched_nms',
    'bbox2roi', 'bbox2offset', 'compute_locations', 'aeloss', 'bmn_loss',
    'box_area', 'box_iou', 'boxes_for_nms', 'bucket2bbox', 'center_size',
    'grid_anchors', 'crop', 'edge_smoothloss', 'focal_loss',
    'gaussian_focal_loss', 'fcos_matcher', 'index2d', 'intersect', 'jaccard',
    'legacy_bbox2delta', 'margin_loss', 'mask_predictor', 'masks_to_boxes',
    'offset2bbox', 'partialconv2d', 'shift', 'random_sampler',
    'sanitize_coordinates', 'tblr2bbox', 'valid_flags',
    'position_embedding_sine', 'position_embedding_learned', 'msms_clsf',
    'maxiou_matcher_match', 'map_roi_levels'
]

BASE_DIR = Path("/mnt/nvme1n1/chenyuxiao/AIChipBenchmark/operators/speed_test/LongTail-Bench/long_tail_bench/samples")

def replace_in_file(path: Path):
    text = path.read_text()
    if "float32" in text:
        new_text = text.replace("float32", "float16")
        path.write_text(new_text)
        print(f"updated {path}")
    else:
        print(f"skip {path} (no float32)")

def main():
    for case in CASES:
        case_dir = BASE_DIR / case
        if not case_dir.exists():
            print(f"case dir missing: {case}")
            continue
        # 遍历该目录下所有 py 文件
        for py_file in case_dir.rglob("*.py"):
            replace_in_file(py_file)

if __name__ == "__main__":
    main()