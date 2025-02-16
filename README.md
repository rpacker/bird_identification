# Bird Detection System

This project implements a bird detection system using the YOLOv8 object detection model. It processes video frames from MP4 files, identifies birds, and saves the top detections with the highest confidence scores.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd bird-detect-1
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare your video files:**

    Place your MP4 video files in the `source_videos` directory.

2.  **Run the bird detection script:**

    ```bash
    python bird_detect.py
    ```

    The script will process each video file in the `source_videos` directory, detect birds, and save the top 3 detections with the highest confidence scores in the `saved_images` directory.

## Example Output

The script will save the top 3 bird detections as JPG images in the `saved_images` directory. The filenames will follow the format:

`top_<rank>_bird_<timestamp>.jpg`

For example:

`top_1_bird_2025-02-16_13-30-00-123456.jpg`

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.