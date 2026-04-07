import argparse

from dynamic_masker_yolov11 import run_preprocess_pipeline
from longstream.preprocess import default_preprocess_config_path, load_preprocess_config
from longstream.preprocess.config import deep_update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_preprocess_config_path())
    parser.add_argument("--video", default=None)
    parser.add_argument("--normalized-root", default=None)
    parser.add_argument("--final-root", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--image-ext", choices=["png", "jpg", "jpeg"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_preprocess_config(args.config)
    overrides = {"input": {}, "io": {}, "video": {}, "dynamic_filter": {}}
    overrides["input"]["mode"] = "video"
    overrides["dynamic_filter"]["enabled"] = True
    if args.video is not None:
        overrides["input"]["source_path"] = args.video
    if args.normalized_root is not None:
        overrides["io"]["normalized_root"] = args.normalized_root
    if args.final_root is not None:
        overrides["io"]["final_root"] = args.final_root
    if args.scene_name is not None:
        overrides["io"]["scene_name"] = args.scene_name
    if args.camera_id is not None:
        overrides["io"]["camera_id"] = args.camera_id
    if args.image_ext is not None:
        overrides["io"]["image_ext"] = args.image_ext
    if args.overwrite:
        overrides["io"]["overwrite"] = True
    if args.target_fps is not None:
        overrides["video"]["target_fps"] = args.target_fps
    if args.model_path is not None:
        overrides["dynamic_filter"]["model_path"] = args.model_path
    cfg = deep_update(cfg, overrides)

    summary = run_preprocess_pipeline(cfg)

    dynamic_frames = 0
    dynamic_instances = 0
    for report in summary.get("reports", []):
        if report.get("is_reference_frame"):
            continue
        decisions = report.get("object_decisions", [])
        if any(item.get("is_dynamic", False) for item in decisions):
            dynamic_frames += 1
        dynamic_instances += sum(1 for item in decisions if item.get("is_dynamic", False))

    preview_video = summary.get("preview_video")
    print("[dynamic-mask-test] finished")
    print(f"  scene_name      : {summary['scene_name']}")
    if "num_frames" in summary:
        print(f"  num_frames      : {summary['num_frames']}")
    print(f"  dynamic_frames  : {dynamic_frames}")
    print(f"  dynamic_objects : {dynamic_instances}")
    print(f"  final_root      : {summary.get('final_root', summary.get('masked_root'))}")
    if preview_video:
        print(f"  preview_video   : {preview_video}")


if __name__ == "__main__":
    main()
