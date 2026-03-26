"""
License Plate Recognition — main runner
========================================
Usage:
    python main.py                          # uses test.MOV
    python main.py --video my_video.mp4     # custom video
    python main.py --video 0                # webcam
"""

import argparse
import cv2
from plate_finder import PlateFinder
from ocr import OCR


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="License Plate Recognition")
    parser.add_argument(
        "--video",
        default="test.MOV",
        help="Path to video file, or 0 for webcam (default: test.MOV)"
    )
    parser.add_argument(
        "--model",
        default="model/binary_128_0.50_ver3.pb",
        help="Path to frozen TF model (.pb)"
    )
    parser.add_argument(
        "--labels",
        default="model/binary_128_0.50_labels_ver2.txt",
        help="Path to labels file"
    )
    parser.add_argument(
        "--min-area", type=int, default=4100,
        help="Minimum plate area in pixels (default: 4100)"
    )
    parser.add_argument(
        "--max-area", type=int, default=15000,
        help="Maximum plate area in pixels (default: 15000)"
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    # Allow "0" string to mean webcam index 0
    video_source = int(args.video) if args.video.isdigit() else args.video

    print("[INFO] Initialising PlateFinder …")
    plate_finder = PlateFinder(
        minPlateArea=args.min_area,
        maxPlateArea=args.max_area
    )

    print("[INFO] Loading OCR model …")
    ocr_model = OCR(modelFile=args.model, labelFile=args.labels)

    print(f"[INFO] Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("[ERROR] Cannot open video source. Check the path / index.")
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        frame_count += 1
        cv2.imshow("Original video", frame)

        # Press Q to quit at any time
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("[INFO] User pressed Q — stopping.")
            break

        # ---- Plate detection ----
        possible_plates = plate_finder.find_possible_plates(frame)

        if possible_plates is not None:
            for i, plate_img in enumerate(possible_plates):
                chars_on_plate = plate_finder.char_on_plate[i]
                recognized_plate, char_count = ocr_model.label_image_list(
                    chars_on_plate, image_size=128
                )

                if char_count > 0:
                    print(f"[Frame {frame_count:05d}] Plate detected: {recognized_plate}")

                cv2.imshow("Detected Plate", plate_img)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    #python main.py --video test.MOV   run this for open
#python Main.py --video 0 #this is for run and  close camera press Q in keyboard