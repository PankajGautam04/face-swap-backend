import cv2
import mediapipe as mp
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import uvicorn

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://faceswapmagic.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,  # Support multiple faces
    refine_landmarks=True,
    min_detection_confidence=0.1,  # Lowered to improve detection
    min_tracking_confidence=0.1
)

def preprocess_image(img, max_size=640):
    """Resize the image to a maximum size while maintaining aspect ratio."""
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def get_landmarks(image, face_index=0):
    """Detect facial landmarks for a specific face in the image."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks or len(results.multi_face_landmarks) <= face_index:
        logging.warning(f"No face detected at index {face_index} in the image!")
        return None
    landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                 for landmark in results.multi_face_landmarks[face_index].landmark]
    logging.info(f"Detected {len(landmarks)} landmarks for face index {face_index}")
    return landmarks

def align_faces(source_img, target_img, source_landmarks, target_landmarks):
    """Align the source face to the target face using homography."""
    src_pts = np.float32(source_landmarks)
    tgt_pts = np.float32(target_landmarks)
    matrix, _ = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC)
    aligned_face = cv2.warpPerspective(source_img, matrix, (target_img.shape[1], target_img.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned_face

def seamless_face_swap(target_img, aligned_face, target_landmarks):
    """Blend the aligned source face onto the target image seamlessly."""
    mask = np.zeros_like(target_img[:, :, 0])
    hull = cv2.convexHull(np.array(target_landmarks, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    # Feather the edges of the mask
    # kernel = np.ones((10, 10), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)

    # Create a 3-channel mask for blending
    mask_3d = cv2.merge([mask, mask, mask]) / 255.0

    # Blend the aligned face with the target image
    blended = (aligned_face * mask_3d + target_img * (1 - mask_3d)).astype(np.uint8)

    # Use seamless cloning for better integration
    center = (int(np.mean(hull[:, 0, 0])), int(np.mean(hull[:, 0, 1])))
    result_img = cv2.seamlessClone(blended, target_img, mask, center, cv2.NORMAL_CLONE)
    return result_img

async def process_images(source_img: np.ndarray, target_img: np.ndarray, face_index: int = 0):
    """Process the source and target images to perform face swapping."""
    # Preprocess images (resize if necessary)
    source_img, source_scale = preprocess_image(source_img)
    target_img, target_scale = preprocess_image(target_img)

    # Detect landmarks
    source_landmarks = get_landmarks(source_img)
    target_landmarks = get_landmarks(target_img, face_index)

    if source_landmarks is None or target_landmarks is None:
        raise HTTPException(status_code=400, detail=f"No faces detected at index {face_index} in one or both images!")

    # Align the source face to the target face
    aligned_face = align_faces(source_img, target_img, source_landmarks, target_landmarks)

    # Perform seamless face swapping
    result_img = seamless_face_swap(target_img, aligned_face, target_landmarks)

    # Resize back to original dimensions if scaled down
    if target_scale < 1:
        orig_h, orig_w = target_img.shape[:2]
        result_img = cv2.resize(result_img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    _, buffer = cv2.imencode(".jpg", result_img)
    return BytesIO(buffer.tobytes())

@app.post("/swap-faces/")
async def swap_faces(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    face_index: int = Form(0)  # Extract face_index from FormData
):
    """Swap the face from the source image onto the selected face in the target image."""
    try:
        logging.info(f"Received swap request with face_index: {face_index}")
        if not source.content_type.startswith('image/'):
            logging.error("Invalid file type for source image")
            raise HTTPException(status_code=400, detail="Source must be an image file (e.g., JPG, PNG).")
        if not target.content_type.startswith('image/'):
            logging.error("Invalid file type for target image")
            raise HTTPException(status_code=400, detail="Target must be an image file (e.g., JPG, PNG).")
        if source.size > 5_000_000 or target.size > 5_000_000:
            logging.error("Images too large")
            raise HTTPException(status_code=400, detail="Images too large!")

        source_data = await source.read()
        target_data = await target.read()
        source_img = cv2.imdecode(np.frombuffer(source_data, np.uint8), cv2.IMREAD_COLOR)
        target_img = cv2.imdecode(np.frombuffer(target_data, np.uint8), cv2.IMREAD_COLOR)

        if source_img is None or target_img is None:
            logging.error("Invalid image files")
            raise HTTPException(status_code=400, detail="Invalid image files!")

        result_io = await process_images(source_img, target_img, face_index)

        logging.info("Face swap completed successfully")
        return StreamingResponse(result_io, media_type="image/jpeg", headers={"Content-Disposition": "attachment; filename=swapped_face_result.jpg"})
    except Exception as e:
        logging.error(f"Error swapping faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face swap failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
