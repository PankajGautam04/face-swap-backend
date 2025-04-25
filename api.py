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
    allow_origins=["https://faceswapmagic.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

def get_landmarks(image, face_index=0):
    """Detects facial landmarks in an image for a specific face."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks or len(results.multi_face_landmarks) <= face_index:
        return None
    landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                 for landmark in results.multi_face_landmarks[face_index].landmark]
    return landmarks

def align_faces(source_img, target_img, source_landmarks, target_landmarks):
    """Warps the source face to align with the target face."""
    hull_source = cv2.convexHull(np.array(source_landmarks, dtype=np.int32))
    hull_target = cv2.convexHull(np.array(target_landmarks, dtype=np.int32))

    rect = cv2.boundingRect(hull_source)
    subdiv = cv2.Subdiv2D(rect)
    for point in source_landmarks:
        subdiv.insert(point)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    source_triangles = []
    target_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        idx1 = source_landmarks.index([pt1[0], pt1[1]]) if [pt1[0], pt1[1]] in source_landmarks else -1
        idx2 = source_landmarks.index([pt2[0], pt2[1]]) if [pt2[0], pt2[1]] in source_landmarks else -1
        idx3 = source_landmarks.index([pt3[0], pt3[1]]) if [pt3[0], pt3[1]] in source_landmarks else -1
        if idx1 != -1 and idx2 != -1 and idx3 != -1:
            source_triangles.append([idx1, idx2, idx3])
            target_triangles.append([idx1, idx2, idx3])

    warped_img = np.zeros_like(target_img)
    for i in range(len(source_triangles)):
        src_pts = np.float32([source_landmarks[source_triangles[i][0]],
                              source_landmarks[source_triangles[i][1]],
                              source_landmarks[source_triangles[i][2]]])
        dst_pts = np.float32([target_landmarks[target_triangles[i][0]],
                              target_landmarks[target_triangles[i][1]],
                              target_landmarks[target_triangles[i][2]]])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        warped_patch = cv2.warpAffine(source_img, M, (target_img.shape[1], target_img.shape[0]))
        mask = np.zeros((target_img.shape[0], target_img.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array([dst_pts[0], dst_pts[1], dst_pts[2]], dtype=np.int32), 255)
        warped_img = cv2.bitwise_and(warped_img, warped_img, mask=cv2.bitwise_not(mask))
        warped_img = cv2.bitwise_or(warped_img, cv2.bitwise_and(warped_patch, warped_patch, mask=mask))
    return warped_img

def seamless_face_swap(target_img, warped_face, target_landmarks):
    """Blends the warped face into the target image."""
    mask = np.zeros_like(target_img[:, :, 0])
    hull = cv2.convexHull(np.array(target_landmarks, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    mask_3d = cv2.merge([mask, mask, mask]) / 255.0

    blended = (warped_face * mask_3d + target_img * (1 - mask_3d)).astype(np.uint8)
    center = (int(np.mean(hull[:, 0, 0])), int(np.mean(hull[:, 0, 1])))
    result_img = cv2.seamlessClone(blended, target_img, mask.astype(np.uint8) * 255, center, cv2.NORMAL_CLONE)
    return result_img

@app.post("/swap-faces/")
async def swap_faces(source: UploadFile = File(...), target: UploadFile = File(...), face_index: int = 0):
    """Swaps the face from the source image onto the selected face in the target image."""
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

        source_landmarks = get_landmarks(source_img)
        if source_landmarks is None:
            logging.error("No face detected in source image")
            raise HTTPException(status_code=400, detail="No face detected in the source image!")

        target_landmarks = get_landmarks(target_img, face_index)
        if target_landmarks is None:
            logging.error(f"No face detected at index {face_index} in target image")
            raise HTTPException(status_code=400, detail=f"No face detected at index {face_index} in the target image!")

        warped_face = align_faces(source_img, target_img, source_landmarks, target_landmarks)
        result_img = seamless_face_swap(target_img, warped_face, target_landmarks)

        _, buffer = cv2.imencode(".jpg", result_img)
        logging.info("Face swap completed successfully")
        return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg", headers={"Content-Disposition": "attachment; filename=swapped_face_result.jpg"})
    except Exception as e:
        logging.error(f"Error swapping faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face swap failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
