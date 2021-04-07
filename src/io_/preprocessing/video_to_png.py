import os

import cv2


def demo_video_to_frames(input_dir: str, output_dir: str) -> None:
    """Converts videos of a directory into their frames (PNG) and saves them to disk.

    :param input_dir: Directory with video files that will be processed.
    :param output_dir: Directory in which frames will be stored.
    """
    for file_name in sorted(os.listdir(input_dir)):
        # navigate to new output directory (per video)
        out_path = os.path.splitext(os.path.join(output_dir, file_name))[0]
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # unpack video
        vid_cap = cv2.VideoCapture(os.path.join(input_dir, file_name))
        frame_read, frame = vid_cap.read()
        frame_count = 0
        while frame_read:
            cv2.imwrite(os.path.join(out_path, "%06d.png" % frame_count), frame)
            frame_count += 1
            frame_read, frame = vid_cap.read()
        vid_cap.release()


if __name__ == '__main__':
    demo_video_to_frames("../../../data/eval/raw", "../../../data/eval/raw_frames")
