import cv2
import os
import time
import numpy as np
import pandas as pd
from deepface import DeepFace
import distance as dst

# --- Configuration ---
DB_PATH = "DB"  # Path to your database directory
PASSED_USERS = ["faris"]  # List of passed user IDs (folder names)
BANNED_USERS = ["rasheed"] # List of banned user IDs (folder names)

# --- Model and Detection Settings ---
# Use a fast detector for real-time. Options: 'opencv', 'ssd', 'mtcnn', 'retinaface', 'yunet'
DETECTOR_BACKEND = 'yunet'
# Choose the recognition model. Ensure it matches how the DB was potentially cached if reusing .pkl
MODEL_NAME = "VGG-Face" # Options: "VGG-Face", "Facenet", "Facenet512", "ArcFace", "SFace", ...
DISTANCE_METRIC = "cosine" # Options: "cosine", "euclidean", "euclidean_l2"

# --- Real-time Settings ---
FRAME_SKIP = 5 # Process every Nth frame to save computation
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LINE_TYPE = cv2.LINE_AA

# --- Colors (BGR format for OpenCV) ---
COLOR_PASSED = (0, 255, 0)  # Green
COLOR_BANNED = (0, 0, 255)  # Red
COLOR_UNKNOWN = (255, 150, 0) # Blue/Cyan
COLOR_NO_MATCH = (255, 255, 255) # White
# --- End Configuration ---

# --- Helper Functions ---

def load_database_embeddings(db_path, model_name, detector_backend, enforce_detection=False, align=True):
    """
    Loads face embeddings from the database into memory.
    Returns a list of dictionaries: [{'identity': str, 'embedding': list}]
    """
    database_embeddings = []
    print(f"Loading database embeddings from: {db_path}")
    start_time = time.time()

    if not os.path.isdir(db_path):
        print(f"ERROR: Database path '{db_path}' not found.")
        return []

    # Check for cached representations file first
    pkl_file = os.path.join(db_path, f"representations_{model_name}.pkl")
    if os.path.exists(pkl_file):
        try:
            print(f"Found cached representations file: {pkl_file}. Loading...")
            df = pd.read_pickle(pkl_file)
            # Convert DataFrame to the desired list format
            for index, row in df.iterrows():
                identity_path = row[f'{model_name}_representation'] # Get embedding
                identity = os.path.basename(os.path.dirname(row['identity'])) # Extract ID from path
                database_embeddings.append({
                    "identity": identity,
                    "embedding": identity_path
                })
            print(f"Loaded {len(database_embeddings)} embeddings from cache.")
            return database_embeddings
        except Exception as e:
            print(f"Warning: Could not load cached file {pkl_file}. Re-processing images. Error: {e}")
            # Fall through to re-processing if cache loading fails

    # If no cache or cache failed, process images
    print("No valid cache found or loading failed. Processing database images...")
    processed_identities = set()

    for identity_name in os.listdir(db_path):
        identity_folder = os.path.join(db_path, identity_name)
        if os.path.isdir(identity_folder):
            for img_name in os.listdir(identity_folder):
                img_path = os.path.join(identity_folder, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Use represent to get embeddings for faces in the image
                        # Note: represent can return multiple embeddings if multiple faces detected
                        embedding_objs = DeepFace.represent(
                            img_path=img_path,
                            model_name=model_name,
                            detector_backend=detector_backend,
                            enforce_detection=enforce_detection, # False allows skipping images with no detectable faces
                            align=align
                        )
                        # Store the embedding for this identity
                        if embedding_objs:
                             # Use the first detected face's embedding for this image
                            embedding = embedding_objs[0]['embedding']
                            database_embeddings.append({
                                "identity": identity_name,
                                "embedding": embedding
                            })
                            processed_identities.add(identity_name)
                            print(f"  Processed: {img_path} -> Identity: {identity_name}")
                        else:
                             print(f"  Warning: No face found or could not represent face in {img_path}")

                    except Exception as e:
                        print(f"  Error processing {img_path}: {e}")

    end_time = time.time()
    print(f"Finished loading database. Found embeddings for {len(processed_identities)} unique identities.")
    print(f"Total embeddings loaded: {len(database_embeddings)}.")
    print(f"Database loading took {end_time - start_time:.2f} seconds.")
    return database_embeddings

def find_threshold(model_name, distance_metric):
    """Gets the verification threshold for a given model and metric."""
    return dst.findThreshold(model_name, distance_metric)

# --- Main Real-time Logic ---

print("Starting real-time face recognition...")

# 1. Load database embeddings
db_embeddings = load_database_embeddings(DB_PATH, MODEL_NAME, DETECTOR_BACKEND)
if not db_embeddings:
    print("Database is empty or failed to load. Exiting.")
    exit()

# 2. Get the verification threshold
threshold = find_threshold(MODEL_NAME, DISTANCE_METRIC)
print(f"Using threshold: {threshold} for {MODEL_NAME} with {DISTANCE_METRIC}")

# 3. Initialize Webcam
cap = cv2.VideoCapture(0) # 0 is usually the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Starting detection loop (press 'q' to quit)...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # DeepFace often prefers RGB
    faces_data = [] # Store data for faces found in this frame cycle

    if frame_count % FRAME_SKIP == 0:
        try:
            # 4. Detect faces in the current frame
            # Returns list of dicts, each containing 'facial_area', 'confidence', etc.
            detected_faces = DeepFace.extract_faces(
                img_path=frame.copy(), # Pass a copy to avoid modification issues
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False, # Don't crash if no face found
                align=True
            )

            # 5. Process each detected face
            for face_info in detected_faces:
                if face_info['confidence'] == 0 and len(detected_faces) == 1:
                    # Sometimes detector returns area [0,0,w,h] if no face, skip this
                    continue

                facial_area = face_info['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                # Crop the face from the original BGR frame for representation
                face_crop = frame[y:y+h, x:x+w]

                if face_crop.size == 0: # Check if crop is valid
                    continue

                status = "ERROR"
                color = COLOR_UNKNOWN
                matched_identity = "Error"

                try:
                    # 6. Generate embedding for the detected face
                    embedding_objs = DeepFace.represent(
                        img_path=face_crop,
                        model_name=MODEL_NAME,
                        enforce_detection=False, # Already detected, but prevents internal error if crop is bad
                        detector_backend='skip' # Skip internal detection
                    )

                    if not embedding_objs:
                         matched_identity = "Rep Fail"
                         status = "ERROR"
                         color = COLOR_UNKNOWN
                    else:
                        embedding = embedding_objs[0]['embedding']

                        # 7. Compare embedding against the database
                        min_dist = float('inf')
                        best_match_identity = None

                        for db_entry in db_embeddings:
                            db_embedding = db_entry['embedding']
                            identity = db_entry['identity']

                            # Calculate distance
                            if DISTANCE_METRIC == 'cosine':
                                dist = dst.findCosineDistance(embedding, db_embedding)
                            elif DISTANCE_METRIC == 'euclidean':
                                dist = dst.findEuclideanDistance(embedding, db_embedding)
                            elif DISTANCE_METRIC == 'euclidean_l2':
                                dist = dst.findEuclideanDistance(dst.l2_normalize(embedding), dst.l2_normalize(db_embedding))
                            else:
                                raise ValueError(f"Unsupported distance metric: {DISTANCE_METRIC}")

                            if dist < min_dist:
                                min_dist = dist
                                best_match_identity = identity

                        # 8. Determine status based on best match and threshold
                        if min_dist <= threshold:
                            matched_identity = best_match_identity
                            if matched_identity in PASSED_USERS:
                                status = "PASSED"
                                color = COLOR_PASSED
                            elif matched_identity in BANNED_USERS:
                                status = "BANNED"
                                color = COLOR_BANNED
                            else:
                                status = "UNKNOWN"
                                color = COLOR_UNKNOWN
                        else:
                            status = "NO MATCH"
                            matched_identity = f"Dist:{min_dist:.2f}" # Show distance if no match
                            color = COLOR_NO_MATCH

                except Exception as e:
                    print(f"Error during representation or comparison: {e}")
                    status = "ERROR"
                    matched_identity = "Proc Error"
                    color = COLOR_UNKNOWN # Use a distinct error color if needed

                # Store results for drawing later
                faces_data.append({
                    "box": (x, y, w, h),
                    "status": status,
                    "identity": matched_identity,
                    "color": color
                })

        except Exception as e:
            # Handle errors during face detection itself
            print(f"Error during face detection: {e}")
            # Optionally draw a general error message on screen


    # 9. Draw bounding boxes and labels (do this every frame using last known data)
    for data in faces_data:
        x, y, w, h = data['box']
        color = data['color']
        label = f"{data['status']}: {data['identity']}"

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        # Draw background rectangle for text
        cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y), color, -1) # Filled rectangle

        # Draw text (use white or black depending on background color for contrast)
        text_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255) # Simple contrast check
        cv2.putText(frame, label, (x, y - baseline), FONT, FONT_SCALE, text_color, FONT_THICKNESS, LINE_TYPE)


    # 10. Display the processed frame
    cv2.imshow('Real-time Face Recognition (Press Q to Quit)', frame)

    frame_count += 1

    # 11. Check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit command received. Stopping...")
        break

# --- Cleanup ---
print("Releasing resources.")
cap.release()
cv2.destroyAllWindows()
print("Done.")