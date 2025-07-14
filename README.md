# Soccer Player Re-identification System

A computer vision system for tracking and re-identifying soccer players in video footage using homography-based 2D ground plane mapping, movement prediction, and proximity-based re-identification.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Files](#model-files)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This system combines multiple computer vision techniques to solve the challenging problem of player re-identification in soccer videos. When players temporarily leave the frame or are occluded, traditional tracking methods lose their identity. This system maintains player identities across such interruptions using:

- **Homography-based 2D mapping**: Transforms pixel coordinates to real-world pitch coordinates
- **Movement prediction**: Uses player trajectory history to predict future positions
- **Proximity-based re-identification**: Matches re-appearing players to lost identities based on predicted positions

## Features

- **Multi-modal Detection**: Supports player, ball, goalkeeper, and referee detection
- **Robust Tracking**: Uses ByteTrack for short-term tracking with custom re-identification for long-term consistency
- **2D Ground Plane Mapping**: Maps video coordinates to real-world pitch coordinates
- **Movement Prediction**: Predicts player positions based on movement history
- **Configurable Parameters**: Easily adjustable thresholds and parameters
- **Real-time Visualization**: Live preview with annotation overlays
- **Multiple Operation Modes**: Player detection, tracking, and specialized analysis modes

## Requirements

### System Requirements
- Python 3.8 or higher
- GPU with CUDA support (recommended) or CPU
- Minimum 8GB RAM (16GB recommended for large videos)
- OpenCV-compatible system

### Python Dependencies
```
opencv-python>=4.5.0
numpy>=1.21.0
supervision>=0.16.0
ultralytics>=8.0.0
tqdm>=4.62.0
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd soccer-player-reidentification
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download model files** (see [Model Files](#model-files) section)
-  

6. **Configure the system** (see [Configuration](#configuration) section)

## Usage

### Basic Usage

```bash
python Re-identification.py --source_video_path "path/to/input/video.mp4" --target_video_path "path/to/output/video.mp4"
```

### Advanced Usage

```bash
python Re-identification.py \
    --source_video_path "input/match.mp4" \
    --target_video_path "output/tracked_match.mp4" \
    --device "cuda" \
    --mode PLAYER_TRACKING
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source_video_path` | str | Required | Path to input video file |
| `--target_video_path` | str | Required | Path for output video file |
| `--device` | str | "cpu" | Device to run models on ("cpu", "cuda") |
| `--mode` | Mode | PLAYER_TRACKING | Operation mode (see modes below) |

### Available Modes

- **PLAYER_DETECTION**: Basic player detection without tracking
- **PLAYER_TRACKING**: Full tracking with re-identification (recommended)

### Adjustable Parameters

Key parameters in `Re-identification.py`:

```python
# Re-identification parameters
reid_distance_threshold = 150  # Distance threshold in ground plane units (cm)
reid_patience = 150           # Frames to remember lost players
STRIDE = 60                   # Periodic cleanup interval

# Tracking parameters
minimum_consecutive_frames = 3  # ByteTrack parameter
history_size = 5               # Movement history for prediction
```

## Model Files

The system requires three pre-trained YOLO models:

1. **Player Detection Model**: `Model/football-player-detection-v9.pt`
2. **Pitch Detection Model**: `data/football-pitch-detection.pt`
3. **Ball Detection Model**: `data/football-ball-detection.pt`

### Model File Structure
```
project/
├── Model/
│   └── football-player-detection-v9.pt
├── data/
│   ├── football-pitch-detection.pt
│   └── football-ball-detection.pt
├── Re-identification.py
└── Config.py
```

### Obtaining Models

Models can be:
- Downloaded from the provided links
- Player Tracking : football-player-detection.pt - "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
- Pitch tracking : football-pitch-detection.pt - "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"


## Output

The system generates:

1. **Annotated Video**: Input video with player tracking annotations

### Annotation Elements

- **Player Bounding Boxes**: Colored ellipses around detected players
- **Player IDs**: Persistent identity numbers (e.g., "#1", "#2")

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'Config'"**
   - Ensure `Config.py` exists in the same directory
   - Verify `SoccerPitchConfiguration` class is properly defined

2. **"Model file not found"**
   - Check model file paths in the code
   - Ensure all three model files are in correct locations

3. **"CUDA out of memory"**
   - Use `--device cpu` for CPU processing
   - Reduce input video resolution
   - Process shorter video segments

4. **Poor tracking performance**
   - Adjust `reid_distance_threshold` for your specific use case
   - Tune `reid_patience` based on video frame rate
   - Calibrate homography points for accurate ground mapping

5. **Homography transformation issues**
   - Manually define source points (`src_pts`) based on your video
   - Ensure destination points (`dst_pts`) match your pitch configuration
   - Use easily identifiable pitch landmarks (corners, penalty boxes)

### Performance Optimization

- Use GPU acceleration when available
- Reduce video resolution for faster processing
- Adjust `STRIDE` parameter for cleanup frequency
- Consider processing video in chunks for very long matches

### Debugging

Enable debug mode by uncommenting the 2D map visualization code to see:
- Ground plane transformations
- Player positions in real-world coordinates
- Movement predictions
- Re-identification decisions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

