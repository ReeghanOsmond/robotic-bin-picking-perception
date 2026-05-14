import subprocess
import sys


EXAMPLES = [
    {"scene_id": 0, "image_id": 0},
    {"scene_id": 0, "image_id": 1},
    {"scene_id": 0, "image_id": 2},
    {"scene_id": 5, "image_id": 0},
    {"scene_id": 5, "image_id": 1},
]


def main() -> None:
    for example in EXAMPLES:
        scene_id = example["scene_id"]
        image_id = example["image_id"]

        print(f"\nRunning scene {scene_id}, image {image_id}")

        command = [
            sys.executable,
            "scripts/analyze_scene_objects.py",
            "--scene-id",
            str(scene_id),
            "--image-id",
            str(image_id),
        ]

        subprocess.run(command, check=True)

    print("\nResults gallery complete.")
    print("Check outputs/figures/ for generated figures.")
    print("Check outputs/reports/ for generated CSV summaries.")


if __name__ == "__main__":
    main()