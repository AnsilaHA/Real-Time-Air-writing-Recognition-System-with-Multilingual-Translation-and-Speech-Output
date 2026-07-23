import cv2
import time
import numpy as np
from hand_tracker import HandTracker
from trajectory_manager import TrajectoryManager

def draw_text_with_bg(img: cv2.Mat, 
                     text: str, 
                     position: tuple, 
                     font: int = cv2.FONT_HERSHEY_SIMPLEX, 
                     scale: float = 0.5, 
                     color: tuple = (255, 255, 255), 
                     thickness: int = 1, 
                     bg_color: tuple = (30, 30, 30), 
                     alpha: float = 0.7) -> cv2.Mat:
    """
    Renders text on the image frame with a semi-transparent background box for readability.
    """
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    
    # Draw background box (handling padding)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 10, y - h - 10), (x + w + 10, y + baseline + 10), bg_color, -1)
    
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Render the text
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return img

def main():
    # Initialize components
    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    trajectory = TrajectoryManager()
    
    # Open webcam capture (using index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set camera resolution (standard 720p if supported, otherwise defaults)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Frame and FPS tracking
    p_time = 0
    writing_mode = True  # True: Writing (drawing), False: Hover (pointer only)
    hand_present_prev = False

    print("=========================================================")
    print("      Air-Writing Hand Tracker & Trajectory Capture       ")
    print("=========================================================")
    print(" Controls:")
    print("   [Space] : Toggle Writing Mode (Drawing / Hover)")
    print("   [C / c] : Clear the Canvas")
    print("   [Q / q] : Quit Application")
    print("=========================================================")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read webcam frame. Exiting...")
            break

        # Mirror the frame horizontally for intuitive air writing
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Detect hands and overlay landmarks/skeletons
        frame, hand_detected = tracker.find_hands(frame, draw=True)

        if hand_detected:
            # Extract position data for Hand 0
            landmarks = tracker.get_landmarks(frame, hand_idx=0)
            
            # Extract Index Finger Tip (Landmark 8)
            index_tip = tracker.get_landmark_by_id(landmarks, 8)
            
            if index_tip:
                px_x, px_y = index_tip["px_x"], index_tip["px_y"]
                norm_x, norm_y, norm_z = index_tip["x"], index_tip["y"], index_tip["z"]

                # If writing_mode is active, capture and append trajectory coordinates
                if writing_mode:
                    trajectory.add_point(px_x, px_y, norm_x, norm_y, norm_z)
                    # Draw a distinct writing tip circle on index finger tip
                    cv2.circle(frame, (px_x, px_y), 8, (0, 0, 255), -1)
                else:
                    # Hovering/Pointer Mode: Draw a green tracking circle
                    cv2.circle(frame, (px_x, px_y), 8, (0, 255, 0), -1)
                    # When hovering, make sure we break consecutive lines when switching back to writing
                    trajectory.trigger_new_stroke()

                # Display index finger coordinates in real-time
                coord_text = f"Fingertip (L8) -> Pixel: ({px_x:4d}, {px_y:4d}) | Norm: ({norm_x:.2f}, {norm_y:.2f}, {norm_z:.2f})"
                draw_text_with_bg(frame, coord_text, (20, h - 30), scale=0.55, color=(0, 255, 255))
            
            hand_present_prev = True
        else:
            # Hand lost: trigger a new stroke so subsequent detections don't draw connections
            if hand_present_prev:
                trajectory.trigger_new_stroke()
                hand_present_prev = False

        # Render current trajectory canvas overlay
        frame = trajectory.draw_trajectory(frame, color=(0, 0, 255), thickness=5)

        # FPS Calculation
        c_time = time.time()
        fps = int(1 / (c_time - p_time)) if (c_time - p_time) > 0 else 0
        p_time = c_time

        # Render Head-Up Display (HUD)
        # 1. Header Bar
        cv2.rectangle(frame, (0, 0), (w, 45), (15, 15, 15), -1)
        cv2.putText(frame, "AIR-WRITING RECOGNITION (PHASE 1)", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 2. Mode indicator (Top Right)
        mode_str = "WRITING" if writing_mode else "HOVER (PAUSED)"
        mode_color = (0, 0, 255) if writing_mode else (0, 255, 0)
        cv2.putText(frame, f"MODE: {mode_str}", (w - 280, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv2.LINE_AA)

        # 3. Stats / Controls overlay panels
        draw_text_with_bg(frame, f"FPS: {fps}", (20, 80), scale=0.5, color=(200, 250, 200))
        draw_text_with_bg(frame, f"Points: {trajectory.total_points}", (20, 115), scale=0.5, color=(200, 200, 250))
        
        # Keyboard controls panel (Bottom Right)
        controls_text = "[Space] Toggle Mode | [C] Clear | [Q] Exit"
        draw_text_with_bg(frame, controls_text, (w - 380, h - 30), scale=0.5, color=(220, 220, 220))

        # Show frame
        cv2.imshow("Real-Time Hand Tracking & Trajectory Capture", frame)

        # Handle keyboard actions
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("Exiting application...")
            break
        elif key == ord('c') or key == ord('C'):
            print("Clearing canvas...")
            trajectory.clear()
        elif key == 32:  # Spacebar code
            writing_mode = not writing_mode
            print(f"Switched mode. Writing Mode: {writing_mode}")
            # Ensure separate stroke is registered when toggling
            trajectory.trigger_new_stroke()

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
