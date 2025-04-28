# Real-Time Field Project (RTFP)

A Python-based video processing system that performs real-time object detection and tracking using computer vision techniques.

## Features

- Object Detection using ORB/SIFT algorithms
- Contour detection and refinement using RDP algorithm
- Object Tracking using KCF and Kalman Filters
- Automatic timestamp recording
- Video clip extraction based on detections
- Cloud-ready architecture

## Requirements

- Python 3.8+
- OpenCV 4.8.0+
- NumPy 1.24.0+
- scikit-image 0.21.0+
- SciPy 1.11.0+
- Pillow 10.0.0+
- python-dotenv 1.0.0+
- boto3 1.28.0+ (for cloud integration)
- tqdm 4.66.0+

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RTFP_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script can be run from the command line:

```bash
python main.py <video_path> <template_path> [--output-dir OUTPUT_DIR]
```

### Arguments:

- `video_path`: Path to the input video file
- `template_path`: Path to the template image file to detect
- `--output-dir`: (Optional) Directory to save output files (default: "output")

### Example:

```bash
python main.py input/video.mp4 input/template.jpg --output-dir results
```

## Output

The script generates:

1. A processed video with:
   - Object detection bounding boxes
   - Tracking visualization
   - Timestamps
   - Object center points

2. Individual video clips for each detection segment

3. Detection timestamps and metadata

## Project Structure

- `main.py`: Main script and command-line interface
- `object_detection.py`: Object detection implementation
- `object_tracking.py`: Object tracking implementation
- `video_processor.py`: Video processing and clip extraction
- `requirements.txt`: Project dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 