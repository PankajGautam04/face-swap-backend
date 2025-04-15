import cv2
import mediapipe as mp
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import uvicorn

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://faceswaptool.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def preprocess_image(img, max_size=640):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def get_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        logging.warning("No faces detected in the image!")
        return None
    landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                 for landmark in results.multi_face_landmarks[0].landmark]
    logging.info(f"Detected {len(landmarks)} landmarks")
    return landmarks

def align_faces(source_img, target_img, source_landmarks, target_landmarks):
    src_pts = np.float32(source_landmarks)
    tgt_pts = np.float32(target_landmarks)
    matrix, _ = cv2.findHomography(src_pts, tgt_pts)
    aligned_face = cv2.warpPerspective(source_img, matrix, (target_img.shape[1], target_img.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned_face

def seamless_face_swap(source_img, target_img, aligned_face, target_landmarks):
    mask = np.zeros_like(target_img[:, :, 0])
    hull = cv2.convexHull(np.array(target_landmarks, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Feather the edges of the mask
    mask = cv2.GaussianBlur(mask, (21, 21), 10)

    center = (int(np.mean(hull[:, 0, 0])), int(np.mean(hull[:, 0, 1])))
    result_img = cv2.seamlessClone(aligned_face, target_img, mask, center, cv2.NORMAL_CLONE)
    return result_img

async def process_images(source_img: np.ndarray, target_img: np.ndarray):
    source_img, source_scale = preprocess_image(source_img)
    target_img, target_scale = preprocess_image(target_img)
    
    source_landmarks = get_landmarks(source_img)
    target_landmarks = get_landmarks(target_img)
    
    if source_landmarks is None or target_landmarks is None:
        raise HTTPException(status_code=400, detail="No faces detected in one or both images!")
    
    aligned_face = align_faces(source_img, target_img, source_landmarks, target_landmarks)
    result_img = seamless_face_swap(source_img, target_img, aligned_face, target_landmarks)
    
    if target_scale < 1:
        orig_h, orig_w = target_img.shape[:2]
        result_img = cv2.resize(result_img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    _, buffer = cv2.imencode(".jpg", result_img)
    return BytesIO(buffer)

@app.post("/swap-faces/")
async def swap_faces(source: UploadFile = File(...), target: UploadFile = File(...)):
    try:
        source_data = await source.read()
        target_data = await target.read()
        
        source_img = cv2.imdecode(np.frombuffer(source_data, np.uint8), cv2.IMREAD_COLOR)
        target_img = cv2.imdecode(np.frombuffer(target_data, np.uint8), cv2.IMREAD_COLOR)
        
        if source_img is None or target_img is None:
            raise HTTPException(status_code=400, detail="Invalid image files!")
        
        result_io = await process_images(source_img, target_img)
        
        return StreamingResponse(result_io, media_type="image/jpeg", headers={"Content-Disposition": "attachment; filename=swapped_face_result.jpg"})
    
    except Exception as e:
        logging.error(f"Error processing images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
