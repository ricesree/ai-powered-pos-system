from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect labeled image data from a webcam."
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Class label to save images under (example: tomato).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of images to save before stopping.",
    )
    parser.add_argument(
        "--output",
        default="dataset",
        help="Base output directory for collected images.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture height (default: 720).",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip frame horizontally for a selfie-style preview.",
    )
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    cleaned = "_".join(label.strip().lower().split())
    return cleaned


def save_frame(frame, output_dir: Path, label: str, image_index: int):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    filename = f"{label}_{timestamp}_{millis:03d}_{image_index:05d}.jpg"
    path = output_dir / filename
    return path, frame


def main() -> int:
    args = parse_args()
    label = sanitize_label(args.label)

    if not label:
        print("Error: --label cannot be empty.")
        return 1

    if args.count <= 0:
        print("Error: --count must be greater than 0.")
        return 1

    try:
        import cv2
    except ImportError:
        print("Error: OpenCV is not installed.")
        print("Install it with: py -m pip install --user opencv-python")
        return 1

    output_dir = Path(args.output) / label
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Error: Could not open camera index {args.camera}.")
        return 1

    print(f"Saving images to: {output_dir.resolve()}")
    print("Controls: [SPACE]=save  [Q]=quit")

    saved = 0

    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            print("Warning: Could not read frame from camera.")
            time.sleep(0.1)
            continue

        if args.flip:
            frame = cv2.flip(frame, 1)

        status = f"Label: {label} | Saved: {saved}/{args.count}"
        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            saved += 1
            image_path, image_data = save_frame(frame, output_dir, label, saved)
            cv2.imwrite(str(image_path), image_data)
            print(f"Saved {saved}/{args.count}: {image_path.name}")

    cap.release()
    cv2.destroyAllWindows()

    if saved == 0:
        print("No images were saved.")
        return 1

    print(f"Done. Saved {saved} image(s) in {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
