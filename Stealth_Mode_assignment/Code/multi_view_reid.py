import argparse
from enum import Enum
from typing import Iterator, List, Dict, Tuple, Optional
import time
import pickle
import json

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont # Import Pillow for image generation

# Import PyTorch for neural network-based feature extraction
import torch
import torch.nn as nn
from torchvision import models, transforms

# Import SoccerPitchConfiguration from Config.py
# Assuming Config.py defines a class like this:
# class SoccerPitchConfiguration:
#     def __init__(self):
#         self.length = 10500  # meters * 100 for cm
#         self.width = 6800    # meters * 100 for cm
#         self.colors = ["#FF0000", "#0000FF", "#FFFF00", "#00FF00"] # Example colors for teams/referees
#         self.pitch_line_color = (255, 255, 255) # White
#         self.background_color = (0, 128, 0) # Green
#         self.center_circle_radius = 915 # cm
#         self.penalty_box_length = 1650 # cm
#         self.penalty_box_width = 4032 # cm (half width of pitch for 18-yard box)
#         self.goal_area_length = 550 # cm
#         self.goal_area_width = 1832 # cm
#         self.goal_width = 732 # cm
#         self.penalty_spot_distance = 1100 # cm from goal line
#         self.map_width_pixels = 800
#         self.map_height_pixels = 500

try:
    from Config import SoccerPitchConfiguratio
except ImportError:
    print("Warning: Config.py not found. Using a default SoccerPitchConfiguration.")
    class SoccerPitchConfiguration:
        def __init__(self):
            self.length = 10500  # cm (105 meters)
            self.width = 6800    # cm (68 meters)
            # Changed default colors to include red for annotations
            self.colors = ["#FF0000", "#FF0000", "#FF0000", "#FF0000"] # All red for annotations
            self.pitch_line_color = (255, 255, 255) # White
            self.background_color = (0, 128, 0) # Green
            self.center_circle_radius = 915 # cm (9.15 meters)
            self.penalty_box_length = 1650 # cm (16.5 meters)
            self.penalty_box_width = 4032 # cm (40.32 meters, half width of pitch for 18-yard box)
            self.goal_area_length = 550 # cm (5.5 meters)
            self.goal_area_width = 1832 # cm (18.32 meters)
            self.goal_width = 732 # cm (7.32 meters)
            self.penalty_spot_distance = 1100 # cm (11 meters from goal line)
            self.map_width_pixels = 800
            self.map_height_pixels = 500


PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, '../Model/football-player-detection-v9.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt') # Not used in this version, but kept for reference
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt') # Not used in this version, but kept for reference

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60 # Process every STRIDE frames for efficiency
CONFIG = SoccerPitchConfiguration()

# Annotator setup for video output
COLORS = CONFIG.colors # This will now be red by default
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

class FeatureExtractor(nn.Module):
    """
    A neural network model for extracting appearance features from player crops.
    Uses a pre-trained ResNet-50 model with the classification layer removed.
    """
    def __init__(self, device='cpu'):
        super(FeatureExtractor, self).__init__()
        # Load a pre-trained ResNet-50 model
        resnet50 = models.resnet50(pretrained=True)
        # Remove the final classification layer to get the feature vector
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        self.features.eval()  # Set the model to evaluation mode
        self.device = device
        self.to(device)
        
        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),  # Standard size for person Re-ID
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_features(self, img_crops: list) -> torch.Tensor:
        """
        Extracts features for a batch of image crops.
        Args:
            img_crops (list of np.ndarray): A list of player image crops from cv2.
        Returns:
            torch.Tensor: A tensor of feature vectors.
        """
        if not img_crops:
            return torch.empty((0, 2048))  # Return empty tensor if no crops

        batch_list = []
        for crop in img_crops:
            # Convert from BGR (cv2) to RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop)
            batch_list.append(self.transform(crop_pil))
            
        batch_tensor = torch.stack(batch_list).to(self.device)
        
        features = self.features(batch_tensor)
        features = features.view(features.size(0), -1)  # Flatten the features
        return features.cpu()

class Mode(Enum):
    PLAYER_MAPPING = 'PLAYER_MAPPING'

def get_pitch_homography(frame_width: int, frame_height: int, config: SoccerPitchConfiguration, video_name: str) -> np.ndarray:
    """
    Computes homography matrix for pitch mapping.
    This function is crucial for accurate ground plane projection.
    For a "perfect outcome," these source points should be precisely calibrated
    using known pitch markings in the video frames. Manual annotation tools
    or more advanced pitch line detection models can be used to obtain
    these points accurately. The current points are examples.
    """
    print(f"\n--- Calibrating Homography for {video_name} ---")
    print("WARNING: Using example calibration points. For high accuracy, replace with actual, precisely measured coordinates for your videos.")

    # Example source points (pixel coordinates in the video frame)
    # These points should correspond to known points on the real football pitch.
    # For instance, the four corners of the pitch, or specific points on the center circle.
    if "broadcast" in video_name.lower():
        src_pts = np.array([
            [frame_width * 0.2, frame_height * 0.9],   # Bottom-left (example)
            [frame_width * 0.8, frame_height * 0.9],   # Bottom-right (example)
            [frame_width * 0.9, frame_height * 0.2],   # Top-right (example)
            [frame_width * 0.1, frame_height * 0.2]    # Top-left (example)
        ], dtype=np.float32)
    elif "tacticam" in video_name.lower():
        src_pts = np.array([
            [frame_width * 0.1, frame_height * 0.85],  # Bottom-left (example)
            [frame_width * 0.8, frame_height * 0.85],  # Bottom-right (example)
            [frame_width * 0.9, frame_height * 0.15],  # Top-right (example)
            [frame_width * 0.05, frame_height * 0.15]  # Top-left (example)
        ], dtype=np.float32)
    else:
        # Default for other videos
        src_pts = np.array([
            [frame_width * 0.2, frame_height * 0.9],
            [frame_width * 0.8, frame_height * 0.9],
            [frame_width * 0.9, frame_height * 0.2],
            [frame_width * 0.1, frame_height * 0.2]
        ], dtype=np.float32)

    # Destination points (coordinates on the 2D ground plane map)
    # These correspond to the real-world dimensions of the pitch.
    # The origin (0,0) is typically one corner of the pitch.
    map_width_pixels, map_height_pixels = config.map_width_pixels, config.map_height_pixels
    
    # Calculate scale factors to map real-world cm to map pixels
    # Assuming origin (0,0) is bottom-left of the pitch in real-world coordinates
    # And map_width_pixels, map_height_pixels are the dimensions of the generated image
    
    # The dst_pts below are set up for a coordinate system where (0,0) is the bottom-left
    # of the pitch, and x increases to the right (length), y increases upwards (width).
    # The generated image will have (0,0) at its top-left.
    # We need to transform real-world (x,y) to image (col, row).
    # col = x * scale_x
    # row = map_height_pixels - (y * scale_y)
    
    # Let's define the destination points based on the pitch dimensions (in cm)
    # For simplicity, let's assume the destination map represents the entire pitch
    # from (0,0) to (config.length, config.width) in real-world cm.
    # We will scale these real-world cm to the map_width_pixels x map_height_pixels.
    
    # Example: Map the pitch corners to the image corners
    # (0,0) -> (0, map_height_pixels)  (Bottom-left of pitch -> Top-left of image)
    # (length, 0) -> (map_width_pixels, map_height_pixels) (Bottom-right -> Top-right)
    # (length, width) -> (map_width_pixels, 0) (Top-right -> Bottom-right)
    # (0, width) -> (0, 0) (Top-left -> Bottom-left)

    # Let's define dst_pts to map the real-world pitch coordinates (0 to length, 0 to width)
    # to the pixel coordinates of our map image (0 to map_width_pixels, 0 to map_height_pixels).
    # We'll assume the map image's (0,0) is top-left, and the pitch's (0,0) is bottom-left.
    # So, a point (x_cm, y_cm) on the pitch maps to (x_pixel, y_pixel) on the image.
    # x_pixel = (x_cm / config.length) * map_width_pixels
    # y_pixel = map_height_pixels - (y_cm / config.width) * map_height_pixels # Invert y-axis for image

    dst_pts = np.array([
        [0, config.map_height_pixels], # Pitch (0,0) -> Image (0, map_height_pixels)
        [config.map_width_pixels, config.map_height_pixels], # Pitch (length,0) -> Image (map_width_pixels, map_height_pixels)
        [config.map_width_pixels, 0], # Pitch (length, width) -> Image (map_width_pixels, 0)
        [0, 0] # Pitch (0, width) -> Image (0,0)
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

def transform_points_to_ground_plane(points_pixel: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
    """Transform pixel coordinates to ground plane coordinates (in cm)."""
    if points_pixel.shape[0] == 0:
        return np.array([])

    points_homogeneous = np.hstack((points_pixel, np.ones((points_pixel.shape[0], 1))))
    transformed_points_homogeneous = (homography_matrix @ points_homogeneous.T).T
    
    # Normalize by the third component to get Cartesian coordinates
    points_ground_pixels = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]

    # Now, convert these pixel coordinates on the map to real-world cm
    # This requires knowing the scale of your map image.
    # Assuming map_width_pixels corresponds to config.length and map_height_pixels to config.width
    
    # Revert the y-axis inversion for actual ground coordinates
    x_cm = (points_ground_pixels[:, 0] / CONFIG.map_width_pixels) * CONFIG.length
    y_cm = (1 - (points_ground_pixels[:, 1] / CONFIG.map_height_pixels)) * CONFIG.width
    
    return np.vstack((x_cm, y_cm)).T

def transform_ground_to_image_pixels(ground_points_cm: np.ndarray, config: SoccerPitchConfiguration) -> np.ndarray:
    """Transform ground plane coordinates (in cm) to image pixel coordinates for the top-down view."""
    if ground_points_cm.shape[0] == 0:
        return np.array([])

    # Scale cm to image pixels
    x_pixels = (ground_points_cm[:, 0] / config.length) * config.map_width_pixels
    # Invert y-axis for image coordinates (0 at top)
    y_pixels = config.map_height_pixels - (ground_points_cm[:, 1] / config.width) * config.map_height_pixels
    
    return np.vstack((x_pixels, y_pixels)).T.astype(int)


def generate_pitch_view_image(frame_idx: int, player_data: List[Tuple[int, np.ndarray, Tuple[int, int, int]]], config: SoccerPitchConfiguration, output_dir: str, video_type: str):
    """
    Generates a top-down image of the football pitch with player positions.
    player_data: List of (global_id, ground_pos_cm, color_rgb_tuple)
    """
    map_width, map_height = config.map_width_pixels, config.map_height_pixels
    image = Image.new("RGB", (map_width, map_height), config.background_color)
    draw = ImageDraw.Draw(image)

    # --- Draw Pitch Lines ---
    line_color = config.pitch_line_color
    line_thickness = 2
    
    # Helper to convert cm to image pixels
    # This function maps real-world (x_cm, y_cm) to image pixel (x_px, y_px)
    # where y_px is inverted (0 at top of image, map_height at bottom)
    def cm_to_img_coords(x_cm, y_cm):
        x_px = (x_cm / config.length) * map_width
        y_px = map_height - (y_cm / config.width) * map_height # Invert y-axis
        return x_px, y_px

    # Outer boundaries
    # Real-world cm: (0,0) is bottom-left, (length, width) is top-right
    # Image pixel: (0,0) is top-left, (map_width, map_height) is bottom-right
    # So, for the rectangle, we need top-left image pixel (0,0) and bottom-right image pixel (map_width, map_height)
    # The coordinates for draw.rectangle are (x0, y0, x1, y1) where (x0, y0) is top-left and (x1, y1) is bottom-right.
    # Since y_px is inverted, a higher real-world Y (like config.width) will result in a smaller y_px.
    # So, for the overall pitch, top-left is (0, config.width) in cm, which maps to (0,0) in image pixels.
    # Bottom-right is (config.length, 0) in cm, which maps to (map_width, map_height) in image pixels.
    draw.rectangle([cm_to_img_coords(0, config.width)[0], cm_to_img_coords(0, config.width)[1],
                    cm_to_img_coords(config.length, 0)[0], cm_to_img_coords(config.length, 0)[1]],
                   outline=line_color, width=line_thickness)

    # Halfway line
    draw.line([cm_to_img_coords(config.length / 2, 0), cm_to_img_coords(config.length / 2, config.width)], fill=line_color, width=line_thickness)

    # Center circle
    center_x, center_y = config.length / 2, config.width / 2
    radius_px_x = (config.center_circle_radius / config.length) * map_width
    radius_px_y = (config.center_circle_radius / config.width) * map_height
    
    center_img_x, center_img_y = cm_to_img_coords(center_x, center_y)
    draw.ellipse([center_img_x - radius_px_x, center_img_y - radius_px_y,
                  center_img_x + radius_px_x, center_img_y + radius_px_y], outline=line_color, width=line_thickness)
    
    # Center spot
    draw.ellipse([center_img_x - 3, center_img_y - 3, center_img_x + 3, center_img_y + 3], fill=line_color)

    # Penalty boxes and goal areas (simplified)
    # For rectangles, (x0, y0, x1, y1) where y0 is top (smaller pixel value) and y1 is bottom (larger pixel value)
    
    # Left goal area
    # Real-world cm: x from 0 to goal_area_length, y from (width - goal_area_width)/2 to (width + goal_area_width)/2
    # Top-left image pixel: (x_px_start, y_px_top) where x_px_start = cm_to_img_coords(0, ...)[0] and y_px_top = cm_to_img_coords(..., (config.width + config.goal_area_width) / 2)[1]
    # Bottom-right image pixel: (x_px_end, y_px_bottom) where x_px_end = cm_to_img_coords(config.goal_area_length, ...)[0] and y_px_bottom = cm_to_img_coords(..., (config.width - config.goal_area_width) / 2)[1]
    
    # Calculate corners for left goal area
    lg_x0, lg_y0_cm = 0, (config.width - config.goal_area_width) / 2
    lg_x1, lg_y1_cm = config.goal_area_length, (config.width + config.goal_area_width) / 2
    lg_px0, lg_py0 = cm_to_img_coords(lg_x0, lg_y1_cm) # Top-left pixel: x0_cm, y1_cm (higher y_cm -> lower y_px)
    lg_px1, lg_py1 = cm_to_img_coords(lg_x1, lg_y0_cm) # Bottom-right pixel: x1_cm, y0_cm (lower y_cm -> higher y_px)
    draw.rectangle([lg_px0, lg_py0, lg_px1, lg_py1], outline=line_color, width=line_thickness)
    
    # Left penalty box
    # Calculate corners for left penalty box
    lp_x0, lp_y0_cm = 0, (config.width - config.penalty_box_width) / 2
    lp_x1, lp_y1_cm = config.penalty_box_length, (config.width + config.penalty_box_width) / 2
    lp_px0, lp_py0 = cm_to_img_coords(lp_x0, lp_y1_cm) # Top-left pixel
    lp_px1, lp_py1 = cm_to_img_coords(lp_x1, lp_y0_cm) # Bottom-right pixel
    draw.rectangle([lp_px0, lp_py0, lp_px1, lp_py1], outline=line_color, width=line_thickness)
    
    # Left penalty spot
    penalty_spot_left_x, penalty_spot_y = config.penalty_spot_distance, config.width / 2
    penalty_spot_left_img_x, penalty_spot_img_y = cm_to_img_coords(penalty_spot_left_x, penalty_spot_y)
    draw.ellipse([penalty_spot_left_img_x - 3, penalty_spot_img_y - 3,
                  penalty_spot_left_img_x + 3, penalty_spot_img_y + 3], fill=line_color)

    # Right goal area
    # Calculate corners for right goal area
    rg_x0_cm, rg_y0_cm = config.length - config.goal_area_length, (config.width - config.goal_area_width) / 2
    rg_x1_cm, rg_y1_cm = config.length, (config.width + config.goal_area_width) / 2
    rg_px0, rg_py0 = cm_to_img_coords(rg_x0_cm, rg_y1_cm) # Top-left pixel
    rg_px1, rg_py1 = cm_to_img_coords(rg_x1_cm, rg_y0_cm) # Bottom-right pixel
    draw.rectangle([rg_px0, rg_py0, rg_px1, rg_py1], outline=line_color, width=line_thickness)
    
    # Right penalty box
    # Calculate corners for right penalty box
    rp_x0_cm, rp_y0_cm = config.length - config.penalty_box_length, (config.width - config.penalty_box_width) / 2
    rp_x1_cm, rp_y1_cm = config.length, (config.width + config.penalty_box_width) / 2
    rp_px0, rp_py0 = cm_to_img_coords(rp_x0_cm, rp_y1_cm) # Top-left pixel
    rp_px1, rp_py1 = cm_to_img_coords(rp_x1_cm, rp_y0_cm) # Bottom-right pixel
    draw.rectangle([rp_px0, rp_py0, rp_px1, rp_py1], outline=line_color, width=line_thickness)
    
    # Right penalty spot
    penalty_spot_right_x = config.length - config.penalty_spot_distance
    penalty_spot_right_img_x, _ = cm_to_img_coords(penalty_spot_right_x, penalty_spot_y)
    draw.ellipse([penalty_spot_right_img_x - 3, penalty_spot_img_y - 3,
                  penalty_spot_right_img_x + 3, penalty_spot_img_y + 3], fill=line_color)

    # Goals (simplified as lines)
    goal_left_y_start = (config.width - config.goal_width) / 2
    goal_left_y_end = (config.width + config.goal_width) / 2
    draw.line([cm_to_img_coords(0, goal_left_y_start), cm_to_img_coords(0, goal_left_y_end)], fill=line_color, width=line_thickness * 2)
    draw.line([cm_to_img_coords(config.length, goal_left_y_start), cm_to_img_coords(config.length, goal_left_y_end)], fill=line_color, width=line_thickness * 2)


    # --- Draw Players ---
    player_radius = 10 # pixels
    try:
        font = ImageFont.truetype("arial.ttf", 12) # Use a common font
    except IOError:
        font = ImageFont.load_default() # Fallback to default font

    for global_id, ground_pos_cm, player_color_rgb in player_data:
        # Convert ground position (cm) to image pixel coordinates
        player_img_x, player_img_y = cm_to_img_coords(ground_pos_cm[0], ground_pos_cm[1])

        # Draw player circle
        draw.ellipse([player_img_x - player_radius, player_img_y - player_radius,
                      player_img_x + player_radius, player_img_y + player_radius],
                     fill=player_color_rgb, outline=(255,255,255), width=1) # White outline

        # Draw player ID
        text_bbox = draw.textbbox((0,0), str(global_id), font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = player_img_x - text_width / 2
        text_y = player_img_y - text_height / 2
        draw.text((text_x, text_y), str(global_id), fill=(0,0,0), font=font) # Black text

    # Save the image
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{video_type}_pitch_view_frame_{frame_idx:06d}.png")
    image.save(output_filename)
    # print(f"Saved pitch view image: {output_filename}")


class PlayerState:
    """Enhanced player state with position history, appearance features, and global ID tracking."""
    def __init__(self, local_id: int, initial_ground_pos: np.ndarray, frame_idx: int, history_size: int = 10):
        self.local_id = local_id
        self.global_id: int = -1 # Global ID for re-identification across videos
        self.last_seen_frame = frame_idx
        self.ground_position_history = [initial_ground_pos.copy()] # Store ground positions in cm
        self.history_size = history_size
        self.velocity_history = [] # Store velocities in cm/frame
        self.avg_position = initial_ground_pos.copy()
        self.confidence_score = 1.0 # Confidence in the assigned global ID
        
        # Neural network feature vectors for appearance-based matching
        self.feature_gallery = [] # List of feature tensors from the neural network
        self.max_feature_gallery_size = 5 # Keep a limited number of feature vectors
        self.avg_feature = None # Average feature vector for quick matching

    def update_position(self, new_ground_pos: np.ndarray, frame_idx: int):
        """Update position with velocity calculation."""
        if len(self.ground_position_history) > 0:
            velocity = new_ground_pos - self.ground_position_history[-1]
            self.velocity_history.append(velocity)
            if len(self.velocity_history) > self.history_size:
                self.velocity_history.pop(0)

        self.ground_position_history.append(new_ground_pos.copy())
        if len(self.ground_position_history) > self.history_size:
            self.ground_position_history.pop(0)
        
        # Update average position
        self.avg_position = np.mean(self.ground_position_history, axis=0)
        self.last_seen_frame = frame_idx

    def update_appearance_feature(self, feature_vector: torch.Tensor):
        """Update the appearance feature gallery with a new feature vector."""
        if feature_vector is None:
            return
            
        # Add new feature to gallery
        self.feature_gallery.append(feature_vector)
        
        # Keep gallery size limited
        if len(self.feature_gallery) > self.max_feature_gallery_size:
            self.feature_gallery.pop(0)
            
        # Update average feature
        if self.feature_gallery:
            self.avg_feature = torch.mean(torch.stack(self.feature_gallery), dim=0)

    def predict_position(self, future_frames: int = 1) -> np.ndarray:
        """Predict future position based on average velocity."""
        if len(self.velocity_history) < 2:
            return self.ground_position_history[-1] # Not enough data for prediction
        
        # Use a few recent velocities for a more stable prediction
        avg_velocity = np.mean(self.velocity_history[-min(len(self.velocity_history), 5):], axis=0)
        predicted_pos = self.ground_position_history[-1] + avg_velocity * future_frames
        return predicted_pos

    def get_movement_pattern(self) -> Dict:
        """Get movement characteristics for matching."""
        if len(self.ground_position_history) < 2:
            return {
                "speed": 0.0,
                "direction": np.array([0.0, 0.0]),
                "stability": 1.0, # High stability if no movement
                "avg_position": self.avg_position
            }
        
        positions = np.array(self.ground_position_history)
        velocities = np.diff(positions, axis=0) # Calculate velocities between consecutive points
        
        avg_speed = np.mean(np.linalg.norm(velocities, axis=1)) # Average magnitude of velocity
        
        # Average direction vector
        avg_direction = np.mean(velocities, axis=0)
        norm_avg_direction = np.linalg.norm(avg_direction)
        if norm_avg_direction > 0:
            avg_direction = avg_direction / norm_avg_direction # Normalize to unit vector
        else:
            avg_direction = np.array([0.0, 0.0]) # No clear direction if no movement

        # Stability based on variance of speed or direction change
        # Lower variance in speed/direction means higher stability
        speed_variance = np.var(np.linalg.norm(velocities, axis=1))
        stability = 1.0 / (1.0 + speed_variance) # Inverse relationship: higher variance -> lower stability
        
        return {
            "speed": avg_speed,
            "direction": avg_direction,
            "stability": stability,
            "avg_position": self.avg_position
        }
        
    def get_appearance_similarity(self, query_feature: torch.Tensor) -> float:
        """
        Calculate appearance similarity between this player and a query feature.
        Returns a similarity score between 0 and 1, where 1 is most similar.
        """
        if self.avg_feature is None or query_feature is None:
            return 0.0
            
        # Compute cosine similarity between average feature and query feature
        cos_sim = torch.nn.functional.cosine_similarity(
            self.avg_feature.unsqueeze(0), 
            query_feature.unsqueeze(0)
        ).item()
        
        # Convert from [-1, 1] range to [0, 1] range
        return (cos_sim + 1) / 2.0

class PositionMap:
    """Stores and manages position data and appearance features for player matching across frames."""
    def __init__(self):
        # global_id -> List of (frame_idx, position_cm, movement_pattern_dict)
        self.player_trajectories: Dict[int, List[Tuple[int, np.ndarray, Dict]]] = {}
        # Spatial grid for fast lookup of players in a given area
        self.position_grid: Dict[Tuple[int, int], List[Tuple[int, int, np.ndarray]]] = {} # (grid_x, grid_y) -> List of (global_id, frame_idx, position_cm)
        self.grid_size = 200 # cm per grid cell (e.g., 2 meters)
        
        # Store appearance features for each global_id
        self.player_features: Dict[int, List[torch.Tensor]] = {}
        self.player_avg_features: Dict[int, torch.Tensor] = {}
        
    def add_player_position(self, global_id: int, frame_idx: int, position_cm: np.ndarray, movement_pattern: Dict):
        """Add position and movement pattern data for a player to the map."""
        if global_id not in self.player_trajectories:
            self.player_trajectories[global_id] = []
        
        self.player_trajectories[global_id].append((frame_idx, position_cm.copy(), movement_pattern.copy()))
        
        # Add to spatial grid for efficient proximity queries
        grid_x, grid_y = int(position_cm[0] // self.grid_size), int(position_cm[1] // self.grid_size)
        if (grid_x, grid_y) not in self.position_grid:
            self.position_grid[(grid_x, grid_y)] = []
        self.position_grid[(grid_x, grid_y)].append((global_id, frame_idx, position_cm))
        
    def add_player_feature(self, global_id: int, feature: torch.Tensor):
        """Add appearance feature for a player."""
        if feature is None:
            return
            
        if global_id not in self.player_features:
            self.player_features[global_id] = []
            
        self.player_features[global_id].append(feature)
        
        # Keep only the most recent features (max 10)
        if len(self.player_features[global_id]) > 10:
            self.player_features[global_id].pop(0)
            
        # Update average feature
        self.player_avg_features[global_id] = torch.mean(torch.stack(self.player_features[global_id]), dim=0)
        
    def get_appearance_similarity(self, global_id: int, query_feature: torch.Tensor) -> float:
        """Calculate appearance similarity between a global_id and query feature."""
        if global_id not in self.player_avg_features or query_feature is None:
            return 0.0
            
        avg_feature = self.player_avg_features[global_id]
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            avg_feature.unsqueeze(0),
            query_feature.unsqueeze(0)
        ).item()
        
        # Convert from [-1, 1] range to [0, 1] range
        return (cos_sim + 1) / 2.0

    def get_nearby_players(self, position_cm: np.ndarray, radius_cm: float = 500) -> List[Tuple[int, np.ndarray]]:
        """
        Get players from the position map within a given radius of a query position.
        Returns a list of (global_id, last_known_position_cm).
        """
        grid_x, grid_y = int(position_cm[0] // self.grid_size), int(position_cm[1] // self.grid_size)
        grid_radius = int(radius_cm // self.grid_size) + 1 # Check surrounding grid cells
        
        nearby_players = []
        seen_global_ids = set() # To avoid duplicate players if they appear in multiple cells

        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):  # Fixed: proper range definition
                cell = (grid_x + dx, grid_y + dy)
                if cell in self.position_grid:
                    for global_id, _, pos in self.position_grid[cell]:
                        if global_id not in seen_global_ids:
                            if np.linalg.norm(position_cm - pos) <= radius_cm:
                                nearby_players.append((global_id, pos))
                                seen_global_ids.add(global_id)
        
        return nearby_players

    def get_player_pattern_at_frame(self, global_id: int, frame_idx: int, window_size: int = 10) -> Optional[Dict]:
        """
        Get movement pattern for a player around a specific frame.
        This is useful for matching based on recent history.
        """
        if global_id not in self.player_trajectories:
            return None
        
        # Find the trajectory points around the given frame_idx
        relevant_trajectory = []
        for f, pos, pattern in reversed(self.player_trajectories[global_id]):
            if f <= frame_idx:
                relevant_trajectory.append((f, pos, pattern))
            if len(relevant_trajectory) >= window_size:
                break
        
        if len(relevant_trajectory) < 2:
            # Not enough data for a meaningful pattern
            if relevant_trajectory:
                return relevant_trajectory[0][2] # Return the single pattern if available
            return None
        
        # Reconstruct PlayerState to get a pattern based on a window
        temp_player_state = PlayerState(local_id=-1, initial_ground_pos=relevant_trajectory[-1][1], frame_idx=relevant_trajectory[-1][0])
        for f, pos, _ in reversed(relevant_trajectory[:-1]):
             temp_player_state.update_position(pos, f)
        
        return temp_player_state.get_movement_pattern()


def process_reference_video(video_path: str, device: str, config: SoccerPitchConfiguration, output_image_dir: str) -> Tuple[PositionMap, str]:
    """Process the longer (reference) video and create position map."""
    print(f"\n=== Processing Reference Video: {video_path} ===")
    
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(source_path=video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3) # ByteTrack for robust local tracking
    
    # Initialize neural network feature extractor
    print("Initializing neural network feature extractor...")
    feature_extractor = FeatureExtractor(device=device)
    
    video_filename = os.path.basename(video_path)
    homography_matrix = get_pitch_homography(video_info.width, video_info.height, config, video_filename)
    
    # Output video writer (for annotated video)
    output_video_path = f"reference_annotated_{video_filename}"
    
    active_players: Dict[int, PlayerState] = {} # local_id -> PlayerState object
    tracker_id_to_local_id_map: Dict[int, int] = {} # ByteTrack ID -> our local ID
    next_local_id = 1
    global_id_counter = 1 # Counter for assigning unique global IDs
    position_map = PositionMap() # Stores global player trajectories

    # Parameters for re-identification within the same video (if tracker loses a player)
    reid_threshold_cm = 150  # Max distance in cm for re-identification
    reid_patience_frames = 150 # How many frames to wait before considering a player lost

    # Determine a color for each global ID for the pitch view image
    global_id_colors: Dict[int, Tuple[int, int, int]] = {}
    color_palette_idx = 0
    
    # Control how often we extract features (every N frames) to save computation
    feature_extraction_stride = 10

    with sv.VideoSink(output_video_path, video_info) as sink:
        pbar = tqdm(total=video_info.total_frames, desc="Processing reference video")
        
        for frame_idx, frame in enumerate(frame_generator):
            result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            
            # Filter for players only (assuming PLAYER_CLASS_ID is correct)
            player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
            
            tracked_detections = tracker.update_with_detections(player_detections)
            
            current_pixel_points = tracked_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            current_ground_points_cm = transform_points_to_ground_plane(current_pixel_points, homography_matrix)
            
            # Extract player crops for feature extraction
            player_crops = []
            if frame_idx % feature_extraction_stride == 0:  # Only extract features periodically
                player_crops = [sv.crop_image(frame, xyxy) for xyxy in tracked_detections.xyxy]
                
            # Extract features if we have crops
            player_features = None
            if player_crops:
                try:
                    player_features = feature_extractor.extract_features(player_crops)
                except Exception as e:
                    print(f"Warning: Feature extraction failed: {e}")
                    player_features = None
            
            frame_tracker_id_to_local_id_map = {} # Map for current frame's tracker IDs to local IDs
            
            current_frame_player_data_for_pitch_view = [] # Data for generating pitch view image

            # Process each tracked detection in the current frame
            for i in range(len(tracked_detections)):
                tracker_id = tracked_detections.tracker_id[i]
                ground_pos_cm = current_ground_points_cm[i]
                
                # Get feature for this player if available
                current_feature = None
                if player_features is not None and i < len(player_features):
                    current_feature = player_features[i]
                
                local_id = None
                
                # Case 1: Tracker ID already known (player continues to be tracked)
                if tracker_id in tracker_id_to_local_id_map:
                    local_id = tracker_id_to_local_id_map[tracker_id]
                    if local_id in active_players:
                        active_players[local_id].update_position(ground_pos_cm, frame_idx)
                        # Update appearance feature if available
                        if current_feature is not None:
                            active_players[local_id].update_appearance_feature(current_feature)
                else:
                    # Case 2: New tracker ID, try to re-identify with existing active players
                    best_match_local_id = None
                    min_distance = float('inf')
                    
                    for lid, player_state in active_players.items():
                        # Only consider players not yet matched in this frame and not seen for too long
                        if lid not in frame_tracker_id_to_local_id_map.values() and \
                           (frame_idx - player_state.last_seen_frame < reid_patience_frames):
                            
                            # Predict player's position to account for movement
                            predicted_pos = player_state.predict_position(future_frames=frame_idx - player_state.last_seen_frame)
                            distance = np.linalg.norm(ground_pos_cm - predicted_pos)
                            
                            # If we have appearance features, use them to improve matching
                            appearance_bonus = 0
                            if current_feature is not None and player_state.avg_feature is not None:
                                similarity = player_state.get_appearance_similarity(current_feature)
                                # Convert similarity (0-1) to a distance reduction (higher similarity = lower distance)
                                appearance_bonus = similarity * reid_threshold_cm * 0.5
                            
                            adjusted_distance = distance - appearance_bonus
                            
                            if adjusted_distance < reid_threshold_cm and adjusted_distance < min_distance:
                                min_distance = adjusted_distance
                                best_match_local_id = lid
                    
                    if best_match_local_id is not None:
                        # Re-identified an existing player
                        local_id = best_match_local_id
                        active_players[local_id].update_position(ground_pos_cm, frame_idx)
                        # Update appearance feature if available
                        if current_feature is not None:
                            active_players[local_id].update_appearance_feature(current_feature)
                    else:
                        # Case 3: Truly a new player (or lost player that couldn't be re-identified)
                        local_id = next_local_id
                        next_local_id += 1
                        active_players[local_id] = PlayerState(local_id, ground_pos_cm, frame_idx)
                        # Add appearance feature if available
                        if current_feature is not None:
                            active_players[local_id].update_appearance_feature(current_feature)
                        
                        # Assign a new global ID
                        active_players[local_id].global_id = global_id_counter
                        global_id_counter += 1
                        
                        # Assign a color for the pitch view
                        if active_players[local_id].global_id not in global_id_colors:
                            # Use the first color from CONFIG.colors (which is now red by default)
                            global_id_colors[active_players[local_id].global_id] = tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4) for h in [CONFIG.colors[0]])
                            color_palette_idx += 1 # Increment, but only first color is used for new global IDs
                
                frame_tracker_id_to_local_id_map[tracker_id] = local_id
                
                # Add player's current state to the global position map
                if local_id in active_players:
                    global_id = active_players[local_id].global_id
                    if global_id > 0: # Ensure a global ID is assigned
                        player_pattern = active_players[local_id].get_movement_pattern()
                        position_map.add_player_position(global_id, frame_idx, ground_pos_cm, player_pattern)
                        
                        # Add appearance feature to position map if available
                        if current_feature is not None:
                            position_map.add_player_feature(global_id, current_feature)
                        
                        # Prepare data for pitch view image
                        player_color = global_id_colors.get(global_id, (255, 255, 255)) # Default white if no color assigned
                        current_frame_player_data_for_pitch_view.append((global_id, ground_pos_cm, player_color))
            
            tracker_id_to_local_id_map = frame_tracker_id_to_local_id_map # Update for next frame
            
            # Cleanup inactive players from active_players dictionary
            # This prevents the dictionary from growing indefinitely
            if frame_idx % STRIDE == 0: # Check periodically
                ids_to_remove = [
                    lid for lid, player_state in active_players.items()
                    if lid not in frame_tracker_id_to_local_id_map.values() and
                    frame_idx - player_state.last_seen_frame > reid_patience_frames * 2 # Give more leeway before removal
                ]
                for lid in ids_to_remove:
                    del active_players[lid]
            
            # Annotate frame for video output
            labels = []
            for i in range(len(tracked_detections)):
                tracker_id = tracked_detections.tracker_id[i]
                local_id = frame_tracker_id_to_local_id_map.get(tracker_id)
                global_id = active_players[local_id].global_id if local_id and local_id in active_players else -1
                labels.append(f"ID:{global_id}" if global_id > 0 else "ID:?")
            
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, tracked_detections)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, tracked_detections, labels=labels)
            
            sink.write_frame(annotated_frame)
            pbar.update(1)

            # Generate and save top-down pitch view image
            if frame_idx % STRIDE == 0 or frame_idx == video_info.total_frames - 1: # Generate periodically and at end
                generate_pitch_view_image(frame_idx, current_frame_player_data_for_pitch_view, config, output_image_dir, "reference")
        
        pbar.close()
    
    print(f"Reference video processed. Found {len(position_map.player_trajectories)} unique global players.")
    return position_map, output_video_path

def match_player_to_reference(
    player_state: PlayerState, 
    position_map: PositionMap, 
    current_frame: int, 
    match_threshold_cm: float = 500, # Max distance for initial match consideration
    pattern_weight: float = 0.5, # Weight for movement pattern similarity
    position_weight: float = 1.0, # Weight for current position similarity
    appearance_weight: float = 2.0 # Weight for appearance similarity (higher because it's more discriminative)
) -> Optional[int]:
    """
    Match a player from the secondary video to a global ID in the reference video's position map.
    This function uses position, movement patterns, and appearance features for robust matching.
    """
    if len(player_state.ground_position_history) == 0:
        return None # Cannot match without position data
    
    current_pos_cm = player_state.ground_position_history[-1]
    player_current_pattern = player_state.get_movement_pattern()
    
    # Step 1: Get nearby players from the reference video's position map
    # This acts as a spatial filter to reduce the search space.
    nearby_ref_players = position_map.get_nearby_players(current_pos_cm, radius_cm=match_threshold_cm)
    
    if not nearby_ref_players:
        return None # No potential matches found nearby
    
    best_match_global_id = None
    best_combined_score = float('inf') # Lower score is better

    for global_id, ref_pos_at_query_time in nearby_ref_players:
        # Step 2: Retrieve the movement pattern for the candidate reference player around the current frame
        # This is crucial for comparing movement characteristics, not just static position.
        ref_pattern = position_map.get_player_pattern_at_frame(global_id, current_frame, window_size=player_state.history_size)
        
        if ref_pattern is None:
            continue # Skip if no sufficient pattern data for reference player

        # Calculate similarity scores (lower is better for "distance" metrics)
        
        # Positional similarity: Euclidean distance between current positions
        pos_distance = np.linalg.norm(current_pos_cm - ref_pos_at_query_time)
        
        # Movement pattern similarity:
        # 1. Average position distance over history
        avg_pos_distance = np.linalg.norm(player_current_pattern["avg_position"] - ref_pattern["avg_position"])
        
        # 2. Speed difference
        speed_diff = abs(player_current_pattern["speed"] - ref_pattern["speed"])
        
        # 3. Direction similarity (cosine similarity, 1 for same, -1 for opposite, 0 for perpendicular)
        # We want to maximize dot product, so minimize (1 - dot_product)
        direction_dot_product = np.dot(player_current_pattern["direction"], ref_pattern["direction"])
        direction_dissimilarity = 1 - direction_dot_product # 0 for same direction, 2 for opposite

        # 4. Appearance similarity (using neural network features)
        appearance_similarity = 0.0
        if player_state.avg_feature is not None:
            appearance_similarity = position_map.get_appearance_similarity(global_id, player_state.avg_feature)
            # Convert from similarity (1 is best) to dissimilarity (0 is best) for consistent scoring
            appearance_dissimilarity = 1.0 - appearance_similarity
        else:
            # If no appearance features, use a neutral value
            appearance_dissimilarity = 0.5

        # Combine scores. Weights can be tuned for accuracy.
        # Current position is often the strongest indicator.
        # Movement pattern adds robustness when positions are ambiguous or noisy.
        # Appearance features provide the most discriminative information when available.
        combined_score = (
            pos_distance * position_weight +
            avg_pos_distance * 0.5 +
            speed_diff * 0.2 +
            direction_dissimilarity * 100 + # Scale direction dissimilarity to be comparable to distances
            appearance_dissimilarity * 500 * appearance_weight # Scale appearance to be comparable and weighted higher
        )
        
        # Apply a penalty if stability differs significantly (optional, for future tuning)
        # stability_diff = abs(player_current_pattern["stability"] - ref_pattern["stability"])
        # combined_score += stability_diff * 50

        if combined_score < best_combined_score:
            best_combined_score = combined_score
            best_match_global_id = global_id
    
    # Final check: Is the best match score below the threshold?
    # This threshold determines how "similar" players must be to be considered a match.
    if best_combined_score < match_threshold_cm * (1 + appearance_weight): # Adjust threshold based on weights
        return best_match_global_id
    else:
        return None


def process_secondary_video(video_path: str, device: str, config: SoccerPitchConfiguration, 
                          position_map: PositionMap, output_image_dir: str) -> str:
    """Process the shorter video and match players to reference."""
    print(f"\n=== Processing Secondary Video: {video_path} ===")
    
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(source_path=video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    # Initialize neural network feature extractor
    print("Initializing neural network feature extractor...")
    feature_extractor = FeatureExtractor(device=device)
    
    video_filename = os.path.basename(video_path)
    homography_matrix = get_pitch_homography(video_info.width, video_info.height, config, video_filename)
    
    output_video_path = f"secondary_annotated_{video_filename}"
    
    active_players: Dict[int, PlayerState] = {} # local_id -> PlayerState object
    tracker_id_to_local_id_map: Dict[int, int] = {}
    local_id_to_global_id_map: Dict[int, int] = {} # Our local ID -> matched global ID
    next_local_id = 1

    # Parameters for re-identification within the same video (if tracker loses a player)
    reid_threshold_cm = 150  # Max distance in cm for re-identification
    reid_patience_frames = 150 # How many frames to wait before considering a player lost
    
    # Threshold for matching to the reference video (can be more lenient than reid_threshold)
    match_to_ref_threshold_cm = 600 # A higher threshold to allow for some initial mismatch or noise

    # Determine a color for each global ID for the pitch view
    global_id_colors: Dict[int, Tuple[int, int, int]] = {}
    color_palette_idx = 0
    
    # Control how often we extract features (every N frames) to save computation
    feature_extraction_stride = 10

    with sv.VideoSink(output_video_path, video_info) as sink:
        pbar = tqdm(total=video_info.total_frames, desc="Processing secondary video")
        
        for frame_idx, frame in enumerate(frame_generator):
            result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            
            player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
            
            tracked_detections = tracker.update_with_detections(player_detections)
            
            current_pixel_points = tracked_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            current_ground_points_cm = transform_points_to_ground_plane(current_pixel_points, homography_matrix)
            
            # Extract player crops for feature extraction
            player_crops = []
            if frame_idx % feature_extraction_stride == 0:  # Only extract features periodically
                player_crops = [sv.crop_image(frame, xyxy) for xyxy in tracked_detections.xyxy]
                
            # Extract features if we have crops
            player_features = None
            if player_crops:
                try:
                    player_features = feature_extractor.extract_features(player_crops)
                except Exception as e:
                    print(f"Warning: Feature extraction failed: {e}")
                    player_features = None
            
            frame_tracker_id_to_local_id_map = {}
            
            current_frame_player_data_for_pitch_view = [] # Data for generating pitch view image

            # Process each tracked detection
            for i in range(len(tracked_detections)):
                tracker_id = tracked_detections.tracker_id[i]
                ground_pos_cm = current_ground_points_cm[i]
                
                # Get feature for this player if available
                current_feature = None
                if player_features is not None and i < len(player_features):
                    current_feature = player_features[i]
                
                local_id = None
                
                # Case 1: Tracker ID already known
                if tracker_id in tracker_id_to_local_id_map:
                    local_id = tracker_id_to_local_id_map[tracker_id]
                    if local_id in active_players:
                        active_players[local_id].update_position(ground_pos_cm, frame_idx)
                        # Update appearance feature if available
                        if current_feature is not None:
                            active_players[local_id].update_appearance_feature(current_feature)
                else:
                    # Case 2: New tracker ID, try to re-identify locally within secondary video
                    best_match_local_id = None
                    min_distance = float('inf')
                    
                    for lid, player_state in active_players.items():
                        if lid not in frame_tracker_id_to_local_id_map.values() and \
                           (frame_idx - player_state.last_seen_frame < reid_patience_frames):
                            predicted_pos = player_state.predict_position(future_frames=frame_idx - player_state.last_seen_frame)
                            distance = np.linalg.norm(ground_pos_cm - predicted_pos)
                            
                            # If we have appearance features, use them to improve matching
                            appearance_bonus = 0
                            if current_feature is not None and player_state.avg_feature is not None:
                                similarity = player_state.get_appearance_similarity(current_feature)
                                # Convert similarity (0-1) to a distance reduction (higher similarity = lower distance)
                                appearance_bonus = similarity * reid_threshold_cm * 0.5
                            
                            adjusted_distance = distance - appearance_bonus
                            
                            if adjusted_distance < reid_threshold_cm and adjusted_distance < min_distance:
                                min_distance = adjusted_distance
                                best_match_local_id = lid
                    
                    if best_match_local_id is not None:
                        local_id = best_match_local_id
                        active_players[local_id].update_position(ground_pos_cm, frame_idx)
                        # Update appearance feature if available
                        if current_feature is not None:
                            active_players[local_id].update_appearance_feature(current_feature)
                    else:
                        # Case 3: Truly a new local player
                        local_id = next_local_id
                        next_local_id += 1
                        active_players[local_id] = PlayerState(local_id, ground_pos_cm, frame_idx)
                        # Add appearance feature if available
                        if current_feature is not None:
                            active_players[local_id].update_appearance_feature(current_feature)
                
                frame_tracker_id_to_local_id_map[tracker_id] = local_id
                
                # Try to match this local player to a global ID from the reference video
                # This matching happens continuously as new position data comes in
                if local_id in active_players:
                    current_player_state = active_players[local_id]
                    
                    # Only try to match if we haven't assigned a global ID yet, or if confidence is low
                    # Or periodically re-evaluate if the match is still strong
                    if current_player_state.global_id == -1 or (frame_idx % (STRIDE * 5) == 0): # Re-evaluate every 5 strides
                        matched_global_id = match_player_to_reference(
                            current_player_state, position_map, frame_idx, match_to_ref_threshold_cm
                        )
                        if matched_global_id is not None:
                            # If a match is found, assign/update the global ID
                            current_player_state.global_id = matched_global_id
                            local_id_to_global_id_map[local_id] = matched_global_id
                            
                            # Assign a color for the pitch view
                            if matched_global_id not in global_id_colors:
                                # Use the first color from CONFIG.colors (which is now red by default)
                                global_id_colors[matched_global_id] = tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4) for h in [CONFIG.colors[0]])
                                color_palette_idx += 1 # Increment, but only first color is used for new global IDs
                    
                    # Prepare data for pitch view image
                    global_id_for_pitch_view = current_player_state.global_id
                    player_color = global_id_colors.get(global_id_for_pitch_view, (128, 128, 128)) # Grey if no global ID yet
                    current_frame_player_data_for_pitch_view.append((global_id_for_pitch_view, ground_pos_cm, player_color))

            tracker_id_to_local_id_map = frame_tracker_id_to_local_id_map
            
            # Cleanup inactive players
            if frame_idx % STRIDE == 0:
                ids_to_remove = [
                    lid for lid, player_state in active_players.items()
                    if lid not in frame_tracker_id_to_local_id_map.values() and 
                    frame_idx - player_state.last_seen_frame > reid_patience_frames * 2
                ]
                for lid in ids_to_remove:
                    if lid in local_id_to_global_id_map:
                        del local_id_to_global_id_map[lid]
                    del active_players[lid]
            
            # Annotate frame for video output
            labels = []
            for i in range(len(tracked_detections)):
                tracker_id = tracked_detections.tracker_id[i]
                local_id = frame_tracker_id_to_local_id_map.get(tracker_id)
                global_id = active_players[local_id].global_id if local_id and local_id in active_players else -1
                labels.append(f"ID:{global_id}" if global_id > 0 else "ID:?")
            
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, tracked_detections)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, tracked_detections, labels=labels)
            
            sink.write_frame(annotated_frame)
            pbar.update(1)

            # Generate and save top-down pitch view image
            if frame_idx % STRIDE == 0 or frame_idx == video_info.total_frames - 1:
                generate_pitch_view_image(frame_idx, current_frame_player_data_for_pitch_view, config, output_image_dir, "secondary")
        
        pbar.close()
    
    matched_players_count = len(set(local_id_to_global_id_map.values())) # Count unique global IDs matched
    print(f"Secondary video processed. Matched {matched_players_count} unique players to reference.")
    return output_video_path

def main(broadcast_video_path: str, tacticam_video_path: str, device: str, output_image_dir: str = "pitch_view_images") -> None:
    """Main function for sequential multi-view processing."""
    print("=== Multi-View Player Re-Identification (Sequential Processing) ===")
    
    # Create the output directory for pitch view images if it doesn't exist
    os.makedirs(output_image_dir, exist_ok=True)

    # Determine which video is longer to use as the reference for building the global position map
    video_info_b = sv.VideoInfo.from_video_path(broadcast_video_path)
    video_info_t = sv.VideoInfo.from_video_path(tacticam_video_path)
    
    if video_info_b.total_frames >= video_info_t.total_frames:
        reference_video = broadcast_video_path
        secondary_video = tacticam_video_path
        print(f"Reference (longer): {reference_video} ({video_info_b.total_frames} frames)")
        print(f"Secondary (shorter): {secondary_video} ({video_info_t.total_frames} frames)")
    else:
        reference_video = tacticam_video_path
        secondary_video = broadcast_video_path
        print(f"Reference (longer): {reference_video} ({video_info_t.total_frames} frames)")
        print(f"Secondary (shorter): {secondary_video} ({video_info_b.total_frames} frames)")
    
    # Step 1: Process the reference video to build the comprehensive PositionMap
    # This map contains the trajectories and movement patterns of all players seen in the reference video.
    position_map, reference_output_video = process_reference_video(reference_video, device, CONFIG, output_image_dir)
    
    # Step 2: Process the secondary video and match its players to the global IDs established in Step 1.
    secondary_output_video = process_secondary_video(secondary_video, device, CONFIG, position_map, output_image_dir)
    
    print(f"\n=== Processing Complete ===")
    print(f"Reference video with global IDs: {reference_output_video}")
    print(f"Secondary video with matched global IDs: {secondary_output_video}")
    print(f"Top-down pitch view images saved to: {output_image_dir}")
    print(f"Both videos now have consistent global player IDs based on spatial and movement features!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequential Multi-View Soccer Player Re-Identification')
    parser.add_argument('--broadcast_video_path', type=str, required=True, 
                       help='Path to the broadcast video file')
    parser.add_argument('--tacticam_video_path', type=str, required=True, 
                       help='Path to the tacticam video file')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='Device to run models on (cpu/cuda)')
    parser.add_argument('--output_image_dir', type=str, default='pitch_view_images',
                       help='Directory to save generated pitch view images')
    
    args = parser.parse_args()
    main(
        broadcast_video_path=args.broadcast_video_path,
        tacticam_video_path=args.tacticam_video_path,
        device=args.device,
        output_image_dir=args.output_image_dir
    )
