import argparse

from dynamic_masker_yolov11 import run_preprocess_pipeline
from longstream.preprocess import default_preprocess_config_path, load_preprocess_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_preprocess_config_path())
    args = parser.parse_args()

    cfg = load_preprocess_config(args.config)
    summary = run_preprocess_pipeline(cfg)
    print("[preprocess] finished")
    print(f"  scene_name      : {summary['scene_name']}")
    print(f"  final_root      : {summary.get('final_root', summary.get('masked_root'))}")
    print(f"  dynamic_enabled : {summary.get('dynamic_filter_enabled', False)}")


if __name__ == "__main__":
    main()
