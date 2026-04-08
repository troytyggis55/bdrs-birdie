from ultralytics import YOLO
import cv2 as cv
import os
import numpy as np

# Initialize the model once
# Using the pre-trained model as requested
model = None 
model_number = "01"


def get_labels(frame, model_path=f"CamVision/YoloModels/Model{model_number}/my_model/my_model.pt"):
    """
    Primary Function: Takes a frame and a specific model path inside "YoloModels folder, returns raw detection results.
    """
    global model
    if model is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_path)
        abs_model_path = os.path.join(project_root, model_path)

        print(f"% YOLO: Loading model from {abs_model_path}")
        model = YOLO(abs_model_path, task='detect')
 
    results = model.predict(frame, conf=0.5, verbose=False)
    return results[0]





def annotate_frame(frame, results):
    """
    Annotation Function: Takes the frame and results, returns an annotated frame.
    """
    # .plot() is the simplest way to get the standard YOLO look
    return results.plot()





