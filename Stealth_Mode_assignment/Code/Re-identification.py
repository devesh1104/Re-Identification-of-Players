import argparse
from enum import Enum
from typing import Iterator, List, Dict, Tuple
import time # For slow processing if needed

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# Import SoccerPitchConfiguration from Config.py
from Config import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, '../Model/football-player-detection-v9.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60 # Used for periodic cleanup
CONFIG = SoccerPitchConfiguration() # Load the configuration

# Annotator setup using CONFIG colors and edges
COLORS = CONFIG.colors # Use colors from Config.py
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'), # Example color, can be from CONFIG
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame

# --- Homography Transformation ---
def get_pitch_homography(frame_width: int, frame_height: int, config: SoccerPitchConfiguration) -> np.ndarray:
    """
    Computes a homography matrix to map pixel coordinates to the 2D pitch space defined by Config.

    ⚠️ IMPORTANT: This is a placeholder. You need to define `src_pts` accurately
    based on your video frame and `dst_pts` based on your `Config.py`'s pitch dimensions.
    The `Config.py` defines pitch dimensions in cm, e.g., 12000cm x 7000cm.
    We will map these to a convenient pixel space for the 2D map.
    """
    # Example: Define 4 points from your video frame (src_pts)
    # These should be easily identifiable points on the pitch, e.g., corners of the penalty box
    # or intersection of lines. These are ARBITRARY for demonstration.
    src_pts = np.array([
        [frame_width * 0.2, frame_height * 0.9],   # Bottom-left corner of field in video
        [frame_width * 0.8, frame_height * 0.9],   # Bottom-right corner of field in video
        [frame_width * 0.9, frame_height * 0.2],   # Top-right corner of field in video
        [frame_width * 0.1, frame_height * 0.2]    # Top-left corner of field in video
    ], dtype=np.float32)

    # Corresponding destination points (real-world coordinates on 2D pitch, in pixels for the map)
    # Map the Config.py dimensions (e.g., 12000cm x 7000cm) to our 2D map size (e.g., 800x500 pixels).
    # Assuming (0,0) of the pitch is bottom-left in Config.py's coordinate system.
    # We will scale these to fit a 800x500 map for visualization.
    # Pitch length: config.length (12000cm), Pitch width: config.width (7000cm)
    
    # Let's map config.length to 800 pixels and config.width to 500 pixels for the map.
    # The origin (0,0) of the map will correspond to (0,0) of the pitch in Config.py.
    map_scale_x = 800 / config.length
    map_scale_y = 500 / config.width

    dst_pts = np.array([
        [0 * map_scale_x, 0 * map_scale_y],                                # (0,0)
        [config.length * map_scale_x, 0 * map_scale_y],                    # (length, 0)
        [config.length * map_scale_x, config.width * map_scale_y],         # (length, width)
        [0 * map_scale_x, config.width * map_scale_y]                      # (0, width)
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

def transform_points_to_ground_plane(points_pixel: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
    """
    Transforms 2D pixel coordinates to 2D ground plane coordinates using a homography matrix.
    """
    if points_pixel.shape[0] == 0:
        return np.array([])

    points_homogeneous = np.hstack((points_pixel, np.ones((points_pixel.shape[0], 1))))
    transformed_points_homogeneous = (homography_matrix @ points_homogeneous.T).T
    points_ground = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]
    return points_ground

# --- Player State and Prediction ---
class PlayerState:
    def __init__(self, permanent_id: int, initial_ground_pos: np.ndarray, frame_idx: int, history_size: int = 5):
        self.permanent_id = permanent_id
        self.last_seen_frame = frame_idx
        self.ground_position_history = [initial_ground_pos] # Store recent positions for prediction
        self.history_size = history_size

    def update_position(self, new_ground_pos: np.ndarray, frame_idx: int):
        self.ground_position_history.append(new_ground_pos)
        if len(self.ground_position_history) > self.history_size:
            self.ground_position_history.pop(0) # Keep history size limited
        self.last_seen_frame = frame_idx

    def predict_next_position(self, current_frame_idx: int) -> np.ndarray:
        """
        Predicts the player's next position based on recent movement.
        Simple linear prediction for demonstration.
        """
        if len(self.ground_position_history) < 2:
            return self.ground_position_history[-1] # Not enough history, return last known

        # Calculate average velocity from last two points
        last_pos = self.ground_position_history[-1]
        prev_pos = self.ground_position_history[-2]
        
        # This assumes constant velocity. For more accuracy, consider Kalman filters.
        predicted_pos = last_pos + (last_pos - prev_pos) 
        return predicted_pos

def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Runs player tracking with re-identification using 2D ground plane mapping,
    movement prediction, and proximity.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    # --- Re-identification Data Structures ---
    # Maps permanent_id -> PlayerState object
    active_player_states: Dict[int, PlayerState] = {} 
    # Maps ByteTrack's tracker_id (temporary) to our permanent_id (long-term)
    current_tracker_to_perm_id_map: Dict[int, int] = {} 

    next_permanent_id = 1
    # Max distance in ground plane units (e.g., cm or meters depending on your Config.py scale)
    # TUNE THIS VALUE based on your pitch scale and how close players can be.
    # If Config.py is in cm, then 150cm = 1.5 meters.
    reid_distance_threshold = 150 # Example: 150 cm on the ground plane
    # How many frames to remember a player's last known position before considering them 'forgotten'
    reid_patience = 150 # Frames


    # Initialize Homography Matrix once at the beginning
    homography_matrix = get_pitch_homography(video_info.width, video_info.height, CONFIG)
    if homography_matrix is None:
        print("WARNING: Homography matrix could not be computed. 2D ground mapping will not work correctly.")
        # This is critical, if homography fails, the 2D mapping and re-ID will be inaccurate.
        # You might want to exit or use a fallback.


    # --- Main Loop ---
    for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
        # Optional: Slow down processing for visualization
        # time.sleep(0.01) 

        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        tracked_detections = tracker.update_with_detections(detections)

        # This map will hold tracker_id -> permanent_id mappings for *only* the current frame's active tracks.
        # It's rebuilt each frame to ensure only present tracker_ids are considered.
        frame_tracker_id_to_perm_id_map: Dict[int, int] = {} 

        # Get bottom-center points of current detections for ground plane mapping
        current_pixel_points = tracked_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        # Transform current pixel points to ground plane coordinates
        current_ground_points = transform_points_to_ground_plane(current_pixel_points, homography_matrix)


        # Re-identification logic for current frame's tracked_detections
        for i in range(len(tracked_detections)):
            tracker_id = tracked_detections.tracker_id[i]
            current_ground_pos = current_ground_points[i]

            permanent_id_for_this_tracker = None

            # 1. First, check if ByteTrack has maintained the tracker_id from the previous frame.
            if tracker_id in current_tracker_to_perm_id_map:
                permanent_id_for_this_tracker = current_tracker_to_perm_id_map[tracker_id]
                # Update the state of the existing player
                if permanent_id_for_this_tracker in active_player_states:
                    active_player_states[permanent_id_for_this_tracker].update_position(current_ground_pos, frame_idx)
                else: # Should not happen if logic is correct, but as a safeguard
                    active_player_states[permanent_id_for_this_tracker] = PlayerState(permanent_id_for_this_tracker, current_ground_pos, frame_idx)
            else:
                # 2. If it's a new tracker_id, try to re-identify with a "lost" player based on proximity to predicted position.
                best_match_id = None
                min_distance = float('inf')

                # Iterate through all currently active (including 'lost' but not yet forgotten) player states
                for p_id, p_state in active_player_states.items():
                    # Only consider players who are currently 'lost' (not assigned to a tracker_id in this frame yet)
                    # and are within the reid_patience window.
                    if frame_idx - p_state.last_seen_frame < reid_patience and \
                       p_id not in frame_tracker_id_to_perm_id_map.values(): # Ensure unique assignment
                        
                        # Predict the lost player's current position
                        predicted_pos = p_state.predict_next_position(current_frame_idx=frame_idx)
                        
                        distance = np.linalg.norm(current_ground_pos - predicted_pos)
                        
                        if distance < reid_distance_threshold and distance < min_distance:
                            min_distance = distance
                            best_match_id = p_id
                
                if best_match_id is not None:
                    # Found a match, re-assign the old permanent ID
                    permanent_id_for_this_tracker = best_match_id
                    # Update the state of the re-identified player
                    active_player_states[permanent_id_for_this_tracker].update_position(current_ground_pos, frame_idx)
                else:
                    # No match found, assign a new permanent ID
                    permanent_id_for_this_tracker = next_permanent_id
                    active_player_states[permanent_id_for_this_tracker] = PlayerState(permanent_id_for_this_tracker, current_ground_pos, frame_idx)
                    next_permanent_id += 1

            # Assign the determined permanent ID to the current tracker_id for this frame
            frame_tracker_id_to_perm_id_map[tracker_id] = permanent_id_for_this_tracker

        # Update the main `current_tracker_to_perm_id_map` for use in the next frame
        current_tracker_to_perm_id_map = frame_tracker_id_to_perm_id_map

        # Periodically clean up player states that haven't been seen for too long.
        # This prevents the `active_player_states` dictionary from growing indefinitely.
        if frame_idx % STRIDE == 0: 
            # Identify IDs that are no longer active and have exceeded patience
            # A player is "inactive" if their permanent_id is not in `frame_tracker_id_to_perm_id_map.values()`
            # (meaning they weren't detected by ByteTrack in this frame) AND they haven't been seen for `reid_patience` frames.
            ids_to_remove = [
                p_id for p_id, p_state in active_player_states.items()
                if p_id not in frame_tracker_id_to_perm_id_map.values() and \
                   frame_idx - p_state.last_seen_frame > reid_patience
            ]
            for p_id in ids_to_remove:
                del active_player_states[p_id]


        # Annotation for the video frame
        labels = []
        for i in range(len(tracked_detections)):
            tracker_id = tracked_detections.tracker_id[i]
            permanent_id = current_tracker_to_perm_id_map.get(tracker_id) 
            label = f"#{permanent_id}" if permanent_id is not None else "#?"
            labels.append(label)

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, tracked_detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, tracked_detections, labels=labels)

        # Removed the 2D Ground Map drawing and combining logic
        # target_map_width = int(map_width * (annotated_frame.shape[0] / map_height))
        # resized_ground_map = cv2.resize(ground_map, (target_map_width, annotated_frame.shape[0]))
        # combined_frame = np.hstack((annotated_frame, resized_ground_map))

        yield annotated_frame # Yield only the annotated frame

def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
   
    if mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    
    # Removed the calculation for combined width, now just use the original video width
    # map_width, map_height = 800, 500
    # ground_map_aspect_ratio = map_width / map_height 
    # final_output_width = video_info.width + int(video_info.height * ground_map_aspect_ratio)
    
    # Corrected sv.VideoInfo initialization: Use video_info.width directly
    output_video_info = sv.VideoInfo(
        width=video_info.width,  # Now just the width of the annotated frame
        height=video_info.height,
        fps=video_info.fps,
        total_frames=video_info.total_frames
    )

    with sv.VideoSink(target_video_path, output_video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soccer Player Detection and Tracking with 2D Ground Map Re-identification and Prediction.')
    parser.add_argument('--source_video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--target_video_path', type=str, required=True, help='Path to save the output video file.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the models on (e.g., "cpu", "cuda").')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_TRACKING, choices=list(Mode), 
                        help='Mode of operation: PLAYER_DETECTION or PLAYER_TRACKING. Only PLAYER_TRACKING implements 2D ground mapping.')
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    )