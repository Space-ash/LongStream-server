import argparse

from longstream.preprocess import prepare_images_to_generalizable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        required=True,
        help="Source image directory or a single image file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Prepared meta_root in LongStream generalizable layout.",
    )
    parser.add_argument(
        "--scene-name", default="runtime_scene", help="Prepared scene name."
    )
    parser.add_argument("--camera-id", default="00", help="Prepared camera id.")
    parser.add_argument(
        "--image-ext",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Prepared frame format.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan image files under --src.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the prepared scene if it already exists.",
    )
    args = parser.parse_args()

    result = prepare_images_to_generalizable(
        source_path=args.src,
        prepared_root=args.out,
        scene_name=args.scene_name,
        camera_id=args.camera_id,
        image_ext=args.image_ext,
        overwrite=args.overwrite,
        recursive=args.recursive,
    )
    print(
        "[longstream] prepared images "
        f"scene={result['scene_name']} frames={result['num_frames']} "
        f"image_dir={result['image_dir']}"
    )


if __name__ == "__main__":
    main()
