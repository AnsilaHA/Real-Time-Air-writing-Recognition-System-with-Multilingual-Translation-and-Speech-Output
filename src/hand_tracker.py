import cv2
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, Any

class HandTracker:
    """
    A class to encapsulate MediaPipe hand detection and landmark tracking.
    """
    def __init__(self, 
                 static_image_mode: bool = False, 
                 max_num_hands: int = 1, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.7, 
                 min_tracking_confidence: float = 0.7):
        """
        Initializes the MediaPipe Hands model.
        
        Args:
            static_image_mode: If False, treats the input images as a video stream.
            max_num_hands: Maximum number of hands to detect. Default is 1 for air writing.
            model_complexity: Complexity of the hand landmark model (0 or 1).
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for landmark tracking.
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        self.results = None

    def find_hands(self, img: cv2.Mat, draw: bool = True) -> Tuple[cv2.Mat, bool]:
        """
        Processes an image frame to detect hands and optionally draws landmarks and connections.
        
        Args:
            img: Input image frame from OpenCV (BGR).
            draw: Flag to determine if hand skeletons should be drawn on the frame.
            
        Returns:
            A tuple of (processed_image, hands_detected_boolean).
        """
        # MediaPipe requires RGB images, but OpenCV reads in BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        hands_detected = False
        if self.results.multi_hand_landmarks:
            hands_detected = True
            if draw:
                for hand_landmarks in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style()
                    )
        return img, hands_detected

    def get_landmarks(self, img: cv2.Mat, hand_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Extracts and returns pixel and normalized coordinates of all 21 hand landmarks.
        
        Args:
            img: Input image frame to calculate pixel coordinates.
            hand_idx: Index of the hand (if multiple are detected).
            
        Returns:
            A list of dictionaries containing landmark ID, normalized coordinates, and pixel coordinates.
            Returns an empty list if no hand landmarks are found.
        """
        landmarks_list = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_idx < len(self.results.multi_hand_landmarks):
                hand_lms = self.results.multi_hand_landmarks[hand_idx]
                h, w, c = img.shape
                
                for lm_id, lm in enumerate(hand_lms.landmark):
                    # Compute pixel coordinates
                    px_x, px_y = int(lm.x * w), int(lm.y * h)
                    
                    landmarks_list.append({
                        "id": lm_id,
                        "name": self.mp_hands.HandLandmark(lm_id).name,
                        "x": lm.x,      # normalized x
                        "y": lm.y,      # normalized y
                        "z": lm.z,      # normalized z (depth)
                        "px_x": px_x,   # pixel x
                        "px_y": px_y    # pixel y
                    })
        return landmarks_list

    def get_landmark_by_id(self, landmarks: List[Dict[str, Any]], lm_id: int) -> Optional[Dict[str, Any]]:
        """
        Helper method to retrieve a specific landmark's data by its ID from the landmarks list.
        
        Args:
            landmarks: List of landmark dictionaries returned by get_landmarks.
            lm_id: The integer ID of the landmark (e.g. 8 for INDEX_FINGER_TIP).
            
        Returns:
            The dictionary for the specified landmark ID, or None if not found.
        """
        for lm in landmarks:
            if lm["id"] == lm_id:
                return lm
        return None
