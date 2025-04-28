import cv2
import numpy as np
import argparse 
import os # Ensure os is imported
from object_detection import ObjectDetector 

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Detect an object in a frame/video using a template.')
parser.add_argument('template_path', type=str, help='Path to the template image file.')
parser.add_argument('input_path', type=str, help='Path to the input image or video file.') 
parser.add_argument('--use_orb', action='store_true', help='Use ORB detector instead of SIFT (default).')
parser.add_argument('--output_path', type=str, default=None, help='Optional path to save the output image/video with detection.')
args = parser.parse_args()

# --- Debugging: Print Parsed Arguments ---
print("-" * 20)
print(f"DEBUG (main.py Init): Template Path Arg: {args.template_path}")
print(f"DEBUG (main.py Init): Input Path Arg:    {args.input_path}")
print(f"DEBUG (main.py Init): Use ORB Arg:       {args.use_orb}")
print(f"DEBUG (main.py Init): Output Path Arg:   {args.output_path}")
print("-" * 20)

# --- Initialize Detector ---
print(f"DEBUG (main.py Init): Initializing ObjectDetector (use_sift={not args.use_orb})...")
detector = ObjectDetector(use_sift=not args.use_orb) 
print(f"DEBUG (main.py Init): ObjectDetector Initialized.")

# --- Load Template ---
print(f"DEBUG (main.py Init): Loading template image: {args.template_path}")
template = cv2.imread(args.template_path)
if template is None:
    print(f"FATAL ERROR: Could not load template image from {args.template_path}")
    exit()
print(f"DEBUG (main.py Init): Template image loaded successfully.")

# --- Determine Input Type ---
is_video = args.input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
print(f"DEBUG (main.py Init): Input path identified as {'Video' if is_video else 'Image'}.")

# --- Initialize detection_timestamps list ---
detection_timestamps = [] 

if is_video:
    # ==========================================
    # --- VIDEO PROCESSING START ---
    # ==========================================
    print(f"\nDEBUG (main.py Video): Opening video capture for: {args.input_path}")
    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        print(f"FATAL ERROR: Could not open video file {args.input_path}")
        exit()
    print(f"DEBUG (main.py Video): Video capture opened.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("DEBUG (main.py Video): Warning - Could not get valid FPS from video. Using default 30.")
        fps = 30.0 
    print(f"DEBUG (main.py Video): Video FPS set to: {fps}")

    writer = None
    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"DEBUG (main.py Video): Creating output directory: {output_dir}")
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"ERROR: Could not create output directory {output_dir}: {e}")
                args.output_path = None # Disable saving if dir creation fails
            
        if args.output_path: # Check again in case it was disabled
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Use a common codec like MP4V
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            print(f"DEBUG (main.py Video): Initializing VideoWriter for: {args.output_path} with {frame_width}x{frame_height} @ {fps} FPS")
            writer = cv2.VideoWriter(args.output_path, fourcc, fps, (frame_width, frame_height))
            if not writer.isOpened():
                 print(f"ERROR: Could not open video writer for path {args.output_path}. Saving disabled.")
                 writer = None 
            else:
                print(f"DEBUG (main.py Video): VideoWriter initialized. Output will be saved.")
    else:
        print("DEBUG (main.py Video): No output path specified. Video will not be saved.")


    frame_count = 0
    print("\nDEBUG (main.py Video): --- Entering Video Processing Loop ---") 
    while True:
        print(f"\nDEBUG (main.py Loop): === Frame {frame_count + 1}: Top of loop iteration ===") 
        
        print(f"DEBUG (main.py Loop): Reading frame {frame_count + 1}...")
        ret, frame = cap.read()
        if not ret:
            print(f"DEBUG (main.py Loop): Failed to read frame {frame_count + 1} or end of video reached. Breaking loop.")
            break 
        print(f"DEBUG (main.py Loop): Frame {frame_count + 1} read successfully.")
        
        # Increment frame_count *after* successful read
        frame_count += 1 
        
        print(f"DEBUG (main.py Loop): Frame {frame_count}: Calling detector.detect_object...")
        # Call the detector
        current_detection_result = detector.detect_object(frame, template)
        print(f"DEBUG (main.py Loop): Frame {frame_count}: detector.detect_object returned.")
        
        # --- THOROUGHLY CHECK THE RESULT ---
        print(f"DEBUG (main.py Loop): Frame {frame_count}: Raw detection result dictionary: {current_detection_result}")
        
        detected_value = current_detection_result.get("detected") # Get value safely
        detected_type = type(detected_value)
        print(f"DEBUG (main.py Loop): Frame {frame_count}: Value of 'detected' key: '{detected_value}' (Type: {detected_type})")

        # Explicitly evaluate the boolean value
        is_detected_flag = bool(detected_value) 
        print(f"DEBUG (main.py Loop): Frame {frame_count}: 'detected' key evaluates to boolean: {is_detected_flag}")
        
        # --- CONDITIONAL BLOCK based on detection ---
        if is_detected_flag is True: # Explicit check for True
            print(f"DEBUG (main.py Loop): Frame {frame_count}: +++ Condition MET: Entering Detected Block +++") 
            
            corners_data = current_detection_result.get('corners')
            location_data = current_detection_result.get('location')
            confidence = current_detection_result.get('confidence', 0.0)

            if corners_data is not None and location_data is not None:
                print(f"DEBUG (main.py Loop): Frame {frame_count}: Corners and Location data found.")
                try:
                    # Convert data for drawing
                    corners = corners_data.astype(int)
                    center = tuple(map(int, location_data))
                    print(f"DEBUG (main.py Loop): Frame {frame_count}: Drawing bounding box and center point.")
                    # Draw bounding box
                    cv2.polylines(frame, [corners.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=3) # Red
                    # Draw center point
                    cv2.circle(frame, center, radius=8, color=(0, 255, 255), thickness=-1) # Yellow
                    # Draw text
                    cv2.putText(frame, f"TARGET ({confidence:.2f})", 
                               (center[0]-40, center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Append timestamp
                    current_time = frame_count / fps
                    print(f"DEBUG (main.py Loop): Frame {frame_count}: Appending timestamp {current_time:.2f} to list.")
                    detection_timestamps.append(current_time)
                    print(f"DEBUG (main.py Loop): Frame {frame_count}: Timestamps list is now: {detection_timestamps}")

                except Exception as e:
                    print(f"ERROR (main.py Loop): Frame {frame_count}: Exception during drawing or timestamp append: {e}")
            else:
                print(f"WARNING (main.py Loop): Frame {frame_count}: Detected=True, but 'corners' or 'location' missing in result dictionary.")
        else:
             print(f"DEBUG (main.py Loop): Frame {frame_count}: --- Condition NOT MET: Skipping Detected Block ---")
             # No drawing or timestamp append if not detected
             pass

        # --- Display Frame ---
        print(f"DEBUG (main.py Loop): Frame {frame_count}: Displaying frame in 'Object Detection' window.")
        cv2.imshow("Object Detection", frame) 

        # --- Save Frame ---
        if writer is not None:
            print(f"DEBUG (main.py Loop): Frame {frame_count}: Writing frame to output video file.")
            writer.write(frame)

        # --- Exit Condition ---
        print(f"DEBUG (main.py Loop): Frame {frame_count}: Checking for 'q' key press...")
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            print("\nDEBUG (main.py Loop): 'q' key pressed. Exiting loop.")
            break
        # Add short print if no key press, optional
        # else:
        #     print(f"DEBUG (main.py Loop): Frame {frame_count}: No key press detected.")


    # ==========================================
    # --- AFTER VIDEO LOOP ---
    # ==========================================
    print("\nDEBUG (main.py Video): --- Exited Video Processing Loop ---")
    
    # Release resources
    print("DEBUG (main.py Video): Releasing video capture...")
    cap.release()
    print("DEBUG (main.py Video): Video capture released.")
    if writer is not None:
        print("DEBUG (main.py Video): Releasing video writer...")
        writer.release()
        print(f"DEBUG (main.py Video): Video writer released. Output saved to: {args.output_path}")
    else:
        print("DEBUG (main.py Video): No video writer to release.")
        
    print("DEBUG (main.py Video): Destroying OpenCV windows...")
    cv2.destroyAllWindows()
    print("DEBUG (main.py Video): OpenCV windows destroyed.")

    # Final check of the timestamps list
    print(f"DEBUG (main.py Video): Final detection timestamps list: {detection_timestamps}") 
    
    # Keep clip extraction removed
    print("\nDEBUG (main.py Video): Clip extraction code section remains REMOVED for debugging.")


elif not is_video: 
    # ==========================================
    # --- SINGLE IMAGE PROCESSING START ---
    # ==========================================
    print(f"\nDEBUG (main.py Image): Processing image file: {args.input_path}")
    frame = cv2.imread(args.input_path)
    if frame is None:
        print(f"FATAL ERROR: Could not load input image from {args.input_path}")
        exit()
    print(f"DEBUG (main.py Image): Image loaded successfully.")

    print(f"DEBUG (main.py Image): Calling detector.detect_object...")
    image_detection_result = detector.detect_object(frame, template) 
    print(f"DEBUG (main.py Image): detector.detect_object returned.")
    print(f"DEBUG (main.py Image): Raw detection result dictionary: {image_detection_result}")

    detected_value = image_detection_result.get("detected", False)
    print(f"DEBUG (main.py Image): 'detected' key evaluates to boolean: {bool(detected_value)}")

    if bool(detected_value):
        print(f"DEBUG (main.py Image): +++ Condition MET: Object DETECTED in image! +++")
        
        corners_data = image_detection_result.get('corners')
        location_data = image_detection_result.get('location')
        confidence = image_detection_result.get('confidence', 0.0)

        if corners_data is not None and location_data is not None:
            try:
                corners = corners_data.astype(int)
                center = tuple(map(int, location_data))
                print(f"DEBUG (main.py Image): Drawing bounding box and center point.")
                cv2.polylines(frame, [corners.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=3)
                cv2.circle(frame, center, radius=8, color=(0, 255, 255), thickness=-1) 
                cv2.putText(frame, f"TARGET ({confidence:.2f})", 
                           (center[0]-40, center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e:
                 print(f"ERROR (main.py Image): Exception during drawing: {e}")
        else:
             print(f"WARNING (main.py Image): Detected=True, but 'corners' or 'location' missing.")

        if args.output_path:
             output_dir = os.path.dirname(args.output_path)
             if output_dir and not os.path.exists(output_dir):
                 print(f"DEBUG (main.py Image): Creating output directory: {output_dir}")
                 os.makedirs(output_dir)
             print(f"DEBUG (main.py Image): Saving output image with detection to: {args.output_path}")
             cv2.imwrite(args.output_path, frame)
             
        print(f"DEBUG (main.py Image): Displaying image in 'Detection Result' window.")
        cv2.imshow("Detection Result", frame)
        print(f"DEBUG (main.py Image): Waiting for key press to close window...")
        cv2.waitKey(0) 
        print(f"DEBUG (main.py Image): Key pressed. Destroying window.")
        cv2.destroyAllWindows()
    else:
        print(f"DEBUG (main.py Image): --- Condition NOT MET: Object NOT detected in image. ---")
        if args.output_path:
             output_dir = os.path.dirname(args.output_path)
             if output_dir and not os.path.exists(output_dir):
                  print(f"DEBUG (main.py Image): Creating output directory: {output_dir}")
                  os.makedirs(output_dir)
             print(f"DEBUG (main.py Image): Saving output image (no detection) to: {args.output_path}")
             cv2.imwrite(args.output_path, frame)
        # Optionally show image even if no detection
        # print(f"DEBUG (main.py Image): Displaying image (no detection) in 'No Detection' window.")
        # cv2.imshow("No Detection", frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

print("\nDEBUG (main.py): --- Script finished ---")
