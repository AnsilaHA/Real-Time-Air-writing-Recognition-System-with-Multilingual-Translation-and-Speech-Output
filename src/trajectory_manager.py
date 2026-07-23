import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class TrajectoryManager:
    """
    Manages the captured trajectory points for air writing.
    Supports multi-stroke drawing to prevent connecting disjoint line segments
    when the hand is hidden/lost or when writing multi-stroke letters.
    """
    def __init__(self, max_points_total: int = 5000):
        """
        Initializes the TrajectoryManager.
        
        Args:
            max_points_total: Maximum limit of points stored to avoid memory leakage.
        """
        self.max_points_total = max_points_total
        # strokes is a list of lists of points. Each inner list is a continuous stroke.
        self.strokes: List[List[Dict[str, Any]]] = [[]]
        self.total_points = 0
        self._new_stroke_pending = False
        # completed_characters is a list of completed character trajectories.
        # Each element is a list of strokes (List[List[Dict[str, Any]]]).
        self.completed_characters: List[List[List[Dict[str, Any]]]] = []

    def add_point(self, px_x: int, px_y: int, x: float, y: float, z: float):
        """
        Appends a tracked point to the current stroke.
        
        Args:
            px_x: Pixel x-coordinate.
            px_y: Pixel y-coordinate.
            x: Normalized x-coordinate.
            y: Normalized y-coordinate.
            z: Normalized z-coordinate (depth).
        """
        # If a new stroke is pending, start it before adding the point
        if self._new_stroke_pending:
            if len(self.strokes[-1]) > 0:
                self.strokes.append([])
            self._new_stroke_pending = False

        # Build point dictionary
        point = {
            "px": (px_x, px_y),
            "norm": (x, y, z)
        }

        # Check total points limit
        if self.total_points >= self.max_points_total:
            # Remove the oldest point from the first non-empty stroke
            for stroke in self.strokes:
                if len(stroke) > 0:
                    stroke.pop(0)
                    self.total_points -= 1
                    break
            # Clean up empty strokes at the start (except the last one if it's the only one)
            while len(self.strokes) > 1 and len(self.strokes[0]) == 0:
                self.strokes.pop(0)

        # Add point to the current stroke
        self.strokes[-1].append(point)
        self.total_points += 1

    def trigger_new_stroke(self):
        """
        Sets a flag to start a new stroke when the next point is added.
        This prevents connecting disjoint lines (e.g. when the hand is lost and returns).
        """
        self._new_stroke_pending = True

    def clear(self):
        """
        Resets the entire trajectory history including completed characters.
        """
        self.strokes = [[]]
        self.total_points = 0
        self._new_stroke_pending = False
        self.completed_characters = []

    def clear_current(self):
        """
        Resets only the current active character trajectory.
        """
        self.strokes = [[]]
        self.total_points = 0
        self._new_stroke_pending = False

    def delete_last_character_trajectory(self):
        """
        Removes the last completed character's trajectory strokes.
        """
        if self.completed_characters:
            self.completed_characters.pop()

    def save_current_character(self):
        """
        Saves the current active character strokes to completed_characters and resets current strokes.
        """
        valid_strokes = [stroke for stroke in self.strokes if len(stroke) > 0]
        if valid_strokes:
            self.completed_characters.append(valid_strokes)
        self.clear_current()

    def draw_trajectory(self, img: cv2.Mat, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 4) -> cv2.Mat:
        """
        Draws the captured trajectory strokes onto the given image frame.
        Draws both completed characters of the current word and the current active strokes.
        
        Args:
            img: The OpenCV BGR image frame to draw on.
            color: BGR tuple for the line color (default is red: (0, 0, 255)).
            thickness: Thickness of the drawing lines.
            
        Returns:
            The image frame with the drawing overlay.
        """
        # Draw all completed characters
        for char_strokes in self.completed_characters:
            for stroke in char_strokes:
                if len(stroke) == 1:
                    cv2.circle(img, stroke[0]["px"], thickness, color, -1)
                elif len(stroke) > 1:
                    for i in range(1, len(stroke)):
                        pt1 = stroke[i - 1]["px"]
                        pt2 = stroke[i]["px"]
                        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Draw current active strokes
        for stroke in self.strokes:
            if len(stroke) == 1:
                # If only one point exists in a stroke, draw a small filled circle
                cv2.circle(img, stroke[0]["px"], thickness, color, -1)
            elif len(stroke) > 1:
                # Draw lines between consecutive points
                for i in range(1, len(stroke)):
                    pt1 = stroke[i - 1]["px"]
                    pt2 = stroke[i]["px"]
                    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return img

    def get_all_points(self) -> List[Dict[str, Any]]:
        """
        Flattens and returns all captured points.
        
        Returns:
            A flat list of all point dictionaries.
        """
        all_pts = []
        for stroke in self.strokes:
            all_pts.extend(stroke)
        return all_pts
