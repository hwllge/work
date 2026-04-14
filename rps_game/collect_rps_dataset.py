import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect hand-crop images for RPS model fine-tuning."
    )
    parser.add_argument("--out", default="dataset/rps_aug", help="Output directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--size", type=int, default=224, help="Saved image size")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Auto-capture interval in seconds",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=30,
        help="Extra pixels around detected hand bbox",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="Maximum number of hands to detect",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without imshow/namedWindow (for SSH or no-display environments)",
    )
    parser.add_argument(
        "--label",
        choices=["scissors", "rock", "paper"],
        default="scissors",
        help="Label used in no-gui mode",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=300,
        help="Stop automatically after saving this many images in no-gui mode",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=1.5,
        help="Seconds to wait before capture starts",
    )
    return parser.parse_args()


def make_square_img(img, size):
    h, w = img.shape[:2]
    bg = np.ones((size, size, 3), np.uint8) * 255
    aspect_ratio = h / w

    if aspect_ratio > 1:
        scale = size / h
        resized_w = int(w * scale)
        resized = cv2.resize(img, (resized_w, size))
        pad = (size - resized_w) // 2
        bg[:, pad : pad + resized_w] = resized
    else:
        scale = size / w
        resized_h = int(h * scale)
        resized = cv2.resize(img, (size, resized_h))
        pad = (size - resized_h) // 2
        bg[pad : pad + resized_h, :] = resized

    return bg


def ensure_dirs(base_dir, labels):
    for label in labels:
        (base_dir / label).mkdir(parents=True, exist_ok=True)


def count_images(base_dir, labels):
    counts = {}
    for label in labels:
        label_dir = base_dir / label
        counts[label] = len(
            [
                p
                for p in label_dir.glob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
    return counts


def save_sample(img, base_dir, label, count):
    ts = int(time.time() * 1000)
    filename = f"{label}_{ts}_{count:05d}.jpg"
    path = base_dir / label / filename
    cv2.imwrite(str(path), img)
    return path


def main():
    args = parse_args()

    labels = ["scissors", "rock", "paper"]
    label_keys = {
        ord("1"): "scissors",
        ord("2"): "rock",
        ord("3"): "paper",
        ord("s"): "scissors",
        ord("r"): "rock",
        ord("p"): "paper",
    }

    out_dir = Path(args.out)
    ensure_dirs(out_dir, labels)
    counts = count_images(out_dir, labels)

    detector = HandDetector(maxHands=args.max_hands)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Camera open failed. Check camera index and permissions.")

    selected_label = args.label
    auto_mode = False
    last_save_time = 0.0
    start_time = time.time()
    saved_this_run = 0

    # If GUI is unavailable (ex: SSH), run in auto capture mode.
    if args.no_gui:
        auto_mode = True

    if not args.no_gui:
        cv2.namedWindow("RPS Collector", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        hands, _ = detector.findHands(frame, draw=False)
        crop_square = None
        bbox = None

        if hands:
            x, y, w, h = hands[0]["bbox"]
            x1 = max(0, x - args.offset)
            y1 = max(0, y - args.offset)
            x2 = min(frame.shape[1], x + w + args.offset)
            y2 = min(frame.shape[0], y + h + args.offset)

            if x2 > x1 and y2 > y1:
                bbox = (x1, y1, x2, y2)
                crop = frame[y1:y2, x1:x2]
                crop_square = make_square_img(crop, args.size)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)

        now = time.time()
        can_capture = (now - start_time) >= args.warmup
        if auto_mode and can_capture and crop_square is not None and (now - last_save_time) >= args.interval:
            counts[selected_label] += 1
            save_sample(crop_square, out_dir, selected_label, counts[selected_label])
            last_save_time = now
            saved_this_run += 1

            if args.no_gui and (saved_this_run % 20 == 0 or saved_this_run == args.max_images):
                print(
                    f"[{selected_label}] saved {saved_this_run}/{args.max_images} "
                    f"(total: {counts[selected_label]})"
                )

            if args.no_gui and saved_this_run >= args.max_images:
                print(f"Done: saved {saved_this_run} images for label '{selected_label}'.")
                break

        if not args.no_gui:
            status_lines = [
                f"label: {selected_label}",
                f"mode: {'AUTO' if auto_mode else 'MANUAL'}",
                f"counts - S:{counts['scissors']} R:{counts['rock']} P:{counts['paper']}",
                "keys: 1/2/3 or s/r/p label | space auto | c capture | q quit",
            ]

            y0 = 28
            for idx, line in enumerate(status_lines):
                cv2.putText(
                    frame,
                    line,
                    (12, y0 + idx * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (20, 240, 240),
                    2,
                    cv2.LINE_AA,
                )

            if crop_square is not None:
                thumb = cv2.resize(crop_square, (160, 160))
                frame[12:172, frame.shape[1] - 172 : frame.shape[1] - 12] = thumb
                cv2.rectangle(
                    frame,
                    (frame.shape[1] - 172, 12),
                    (frame.shape[1] - 12, 172),
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("RPS Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in label_keys:
                selected_label = label_keys[key]
            elif key == ord(" "):
                auto_mode = not auto_mode
                last_save_time = 0.0
            elif key == ord("c") and crop_square is not None:
                counts[selected_label] += 1
                save_sample(crop_square, out_dir, selected_label, counts[selected_label])

    cap.release()
    if not args.no_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()