import cv2 as cv
import numpy as np
from CamVision.bucketballsyolo import * 



# Define lower and upper HSV boundaries for the color blue
LOWER_BLUE = np.array([90, 70, 0])
UPPER_BLUE =  np.array([110, 255, 255])


def BlueMask(frame):
    '''
    Input: image in HSV color space
    Output: Blue Mask
    '''
    imgHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    blue_mask = cv.inRange(imgHSV,LOWER_BLUE,UPPER_BLUE)

    # 2. Use a specific kernel size rather than "None"
    kernel = np.ones((5, 5), np.uint8)

    # 3. Use OPENING to remove noise (Erode then Dilate)
    # Only 1 or 2 iterations!
    blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_OPEN, kernel, iterations=1)

    # 4. Use CLOSING to fill the glare holes from your sensors
    # This will merge the "ring" of the ball back into a solid circle
    blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_CLOSE, kernel, iterations=2)

    #blue_mask[550:616, :] = 0

    return blue_mask


# Define lower and upper HSV boundaries for the color red 
# (this motofucka is the only color that expands before zero and after zero, so we need two ranges)
LOWER_RED_RANGE1 = np.array([0, 70, 0])
UPPER_RED_RANGE1 = np.array([5, 255, 255])

LOWER_RED_RANGE2 = np.array([175, 70, 0])
UPPER_RED_RANGE2 = np.array([180, 255, 255])





def RedMask(frame):
    '''
    Input: image in HSV color space
    Output: Blue Mask
    '''

    imgHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # 1. Create the raw masks
    red_mask1 = cv.inRange(imgHSV, LOWER_RED_RANGE1, UPPER_RED_RANGE1)
    red_mask2 = cv.inRange(imgHSV, LOWER_RED_RANGE2, UPPER_RED_RANGE2)
    red_mask = cv.bitwise_or(red_mask1, red_mask2)

    # 2. Use a specific kernel size rather than "None"
    kernel = np.ones((5, 5), np.uint8)

    # 3. Use OPENING to remove noise (Erode then Dilate)
    # Only 1 or 2 iterations!
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel, iterations=1)

    # 4. Use CLOSING to fill the glare holes from your sensors
    # This will merge the "ring" of the ball back into a solid circle
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel, iterations=2)

    return red_mask








def localize_ball_lowest_contour(frame, color='R'):
    """
    Returns: 
      found (bool): True if ball detected
      ball_data (dict): {'center': (x,y), 'rect': (x,y,w,h), 'area': a}
    """
    # 1. Get the polished mask based on color
    if color.upper() == 'B':
        mask = BlueMask(frame)
    else:
        mask = RedMask(frame)

    # 2. Find all contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    found = False
    ball_data = {'center': (0,0), 'rect': (0,0,0,0), 'area': 0}

    if contours:
        # Filter out tiny noise (area < 100 pixels)
        valid_contours = [c for c in contours if cv.contourArea(c) > 100]
        
        if valid_contours:
            # STRATEGY: Select the contour with the lowest Y-coordinate (max Y value)
            # We use the bounding box bottom (y + h) to find the one closest to the robot
            best_cnt = max(valid_contours, key=lambda c: cv.boundingRect(c)[1] + cv.boundingRect(c)[3])
            
            x, y, w, h = cv.boundingRect(best_cnt)
            M = cv.moments(best_cnt)
            
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                found = True
                ball_data = {
                    'center': (cX, cY),
                    'rect': (x, y, w, h),
                    'area': cv.contourArea(best_cnt)
                }
                
    return found, ball_data







def localize_ball_yolo(frame, color="B"):
    """
    Uses YOLO to find the ball. If multiple are found, picks the one 
    lowest in the frame (closest to the robot).
    """
    # 1. Run YOLO inference
    results = get_labels(frame)
    
    # 2. Extract boxes, confidences, and class IDs
    boxes = results.boxes
    found_balls = []

    for box in boxes:
        # Get class name (e.g., 'red_ball' or 'blue_ball')
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]

        # Only process the ball color we are looking for
        if cls_name == color:
            # box.xyxy is [x1, y1, x2, y2]
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords
            w, h = x2 - x1, y2 - y1
            
            ball_info = {
                'center': (int(x1 + w/2), int(y1 + h/2)),
                'rect': (int(x1), int(y1), int(w), int(h)),
                'area': float(w * h),
                'bottom_y': y2  # Used for sorting
            }
            found_balls.append(ball_info)

    # 3. Apply the "Lowest Contour" (Highest Y) strategy
    if found_balls:
        # Sort by bottom_y and pick the largest value
        best_ball = max(found_balls, key=lambda b: b['bottom_y'])
        
        # Clean up the dict to match your standard before returning
        del best_ball['bottom_y']
        return True, best_ball

    return False, None





def track_ball_window(frame, color='R', prev_window=None, margin=30):
    """
    Crops the frame based on the previous window + margin.
    Finds the contour in that ROI closest to the previous center.
    Returns: ball_data dictionary OR None.
    """
    if prev_window is None:
        return None

    # 1. Define ROI (Previous Window + Margin)
    px, py, pw, ph = prev_window
    x1 = max(0, px - margin)
    y1 = max(0, py - margin)
    x2 = min(frame.shape[1], px + pw + margin)
    y2 = min(frame.shape[0], py + ph + margin)

    roi = frame[y1:y2, x1:x2]

    # 2. Process ROI
    if color.upper() == 'B':
        mask = BlueMask(roi)
    else:
        mask = RedMask(roi)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Target: Center of the crop (where the ball was)
    roi_center = (roi.shape[1] // 2, roi.shape[0] // 2)
    best_cnt = None
    min_dist = float('inf')

    for cnt in contours:
        if cv.contourArea(cnt) < 100: continue
        
        M = cv.moments(cnt)
        if M["m00"] == 0: continue
        
        # Local ROI coordinates
        lx, ly = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        dist = (lx - roi_center[0])**2 + (ly - roi_center[1])**2
        
        if dist < min_dist:
            min_dist = dist
            best_cnt = cnt

    # 3. Translate to Global and Return Standard Dictionary
    if best_cnt is not None:
        x, y, w, h = cv.boundingRect(best_cnt)
        M = cv.moments(best_cnt)
        
        # Global Coordinates
        cX = int(M["m10"] / M["m00"]) + x1
        cY = int(M["m01"] / M["m00"]) + y1
        
        ball_data = {
            'center': (cX, cY),
            'rect': (x + x1, y + y1, w, h), # Globalized rectangle
            'area': M["m00"]
        }
        return ball_data

    return None