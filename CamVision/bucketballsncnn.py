import ncnn
import numpy as np
import cv2 as cv
import os

# Global handles
net = None
recorder = None

def get_labels(frame, model_path="CamVision/YoloModels/yolo26n_ncnn_model"):
    """
    Primary Function: Raw NCNN inference. 
    Returns: [class_id, x_center, y_center, width, height, confidence]
    """
    global net
    h, w, _ = frame.shape

    if net is None:
        # Get Absolute Paths
        base_path = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_path)
        abs_folder = os.path.normpath(os.path.join(project_root, model_path))
        
        # Load NCNN Net
        net = ncnn.Net()
        # Optimization: Use Vulkan if you had a GPU, but on Pi we stick to CPU
        net.load_param(os.path.join(abs_folder, "model.ncnn.param"))
        net.load_model(os.path.join(abs_folder, "model.ncnn.bin"))
        print(f"% NCNN: Model loaded from {abs_folder}")

    # 1. Pre-process: Manual resize to 640x640 (Standard YOLO size)
    # This is faster than letting a big library guess how to padding/resize
    img_resized = cv.resize(frame, (640, 640))
    mat_in = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 640, 640)
    
    # 2. Normalize: (x - 0) * (1/255) => puts pixels in 0.0 - 1.0 range
    mat_in.substract_mean_normalize([0,0,0], [1/255.0, 1/255.0, 1/255.0])

    # 3. Extract: "in0" and "out0" are the standard I/O names for YOLOv8 NCNN exports
    with net.create_extractor() as ex:
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")
        output = np.array(mat_out) # Shape: (84, 8400)

    # 4. Post-process: Manual NMS-lite
    detections = []
    for i in range(output.shape[1]):
        scores = output[4:, i]
        conf = np.max(scores)
        if conf > 0.7:  # Higher threshold to kill those "microwaves"
            cls = np.argmax(scores)
            cx, cy, bw, bh = output[:4, i]
            # Normalize coordinates relative to original frame (0.0 to 1.0)
            detections.append([cls, cx/640, cy/640, bw/640, bh/640, conf])
            
    return detections

def annotate_frame(frame, detections):
    """
    Manual drawing since we aren't using results.plot() anymore.
    """
    h, w, _ = frame.shape
    for det in detections:
        cls, cx, cy, bw, bh, conf = det
        # Back to pixels
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, f"ID:{int(cls)} {conf:.2f}", (x1, y1-10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def record_frame(annotated_frame, output_dir="CamVision/Recordings"):
    global recorder
    if recorder is None:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        h, w, _ = annotated_frame.shape
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        # Setting to 5 FPS to be realistic with the Pi's heat
        recorder = cv.VideoWriter(os.path.join(output_dir, 'yolo_mission.avi'), fourcc, 0.5, (w, h))
    recorder.write(annotated_frame)

def close_recorder():
    global recorder
    if recorder is not None:
        recorder.release()
        recorder = None
        print("% NCNN: Recorder closed.")