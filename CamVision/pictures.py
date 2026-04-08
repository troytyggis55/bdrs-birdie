from scam import cam
from uservice import service
from sedge import edge
from sgpio import gpio

import cv2 as cv
import os
import sys
import select

def check_keyboard():
    # Check if there is data
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        # Use readline instead of read(1) to play nice with "Enter"
        line = sys.stdin.readline()
        if len(line) > 0:
            return line[0].lower() # Just give us the first letter
    return None


### DO NOT KILL THE STREAM WHEN TAKING PICTURES
### ONLY KILL STREAM BEFORE TESTING VISUAL SERVOING AND STUFF THAT REQUIRES HIGH FPS
def TakePicture(save, folder="BucketBalls"):
  if cam.useCam:
    ok, img, imgTime = cam.getImage()
    if not ok: # size(img) == 0):
      if cam.imageFailCnt < 5:
        print("% Failed to get image.")
    else:
      h, w, ch = img.shape
      if not service.args.silent:
        # print(f"% At {imgTime}, got image {cam.cnt} of size= {w}x{h}")
        pass
      if not gpio.onPi:
        try:
          cv.imshow('frame for analysis', img)
        except:
          print("% mqtt-client::imageAnalysis: failed to show camera image");
      if save:
        if not os.path.exists(folder):
          os.makedirs(folder)
          print(f"% Created directory: {folder}")
        filename = f"image_{imgTime.strftime('%Y_%b_%d_%H%M%S_')}{cam.cnt:03d}.jpg"
        fn = os.path.join(folder, filename)
        cv.imwrite(fn, img)

        if not service.args.silent:
          print(f"% Saved image {fn}")
      else:
        print("# imageAnalysis:: image not saved")
      pass
    pass
  pass


recorder = None



def record_frame(annotated_frame, output_dir="VisionOutput"):
    """
    Output Logic: Handles saving the frame to a video file.
    """
    global recorder

    if len(annotated_frame.shape) == 2:
        annotated_frame = cv.cvtColor(annotated_frame, cv.COLOR_GRAY2BGR)

    if recorder is None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        h, w, _ = annotated_frame.shape
        path = os.path.join(output_dir, 'record.avi')
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        recorder = cv.VideoWriter(path, fourcc, 8, (w, h))
        print(f"% Recording to {path}")
    
    recorder.write(annotated_frame)


def close_recorder():
    """
    Cleanup: Safely closes the video file.
    """
    global recorder
    if recorder is not None:
        recorder.release()
        recorder = None
        print("% Video recording saved.")