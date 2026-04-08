import time as t
import os
import datetime

from uservice import service
from sedge import edge
from spose import pose
from scam import cam

from CamVision.visualcontrol import *
from CamVision.bucketballsyolo import *
from CamVision.pictures import *
from CamVision.ballcoords import pixels_to_robot_coords

def _log_detections(frame, frame_idx, log_path):
    """
    Scans the full frame for both ball colors, converts detections to robot-frame
    coordinates, and appends one CSV row per found ball to log_path.
    Runs independently of the control loop — safe to call every N frames.
    """
    timestamp = t.time()
    rows = []
    for color in ('B', 'R'):
        found, ball = localize_ball_lowest_contour(frame, color=color)
        if found:
            cX, cY = ball['center']
            x, y, w, h = ball['rect']
            r_px = min(w, h) / 2.0
            coords = pixels_to_robot_coords([(cX, cY, r_px)])
            if coords:
                x_r, y_r, z_r = coords[0]
                rows.append(f"{frame_idx},{timestamp:.3f},{color},{x_r:.4f},{y_r:.4f},{z_r:.4f}\n")
    if rows:
        with open(log_path, 'a') as f:
            f.writelines(rows)


Kp_turn = 0.010  # Adjusts how fast the robot spins to center the ball
Kp_fwd  = 0.020  # Adjusts how fast the robot drives toward the ball
target_y = 430
image_center_x = 410 # resolution of image is 820x616 
Kd_fwd = 0.050

def create_debug_view(frame, ball_data, color, margin=30):
    # 1. Get the mask for the right-side view
    mask = BlueMask(frame) if color == 'B' else RedMask(frame)
    mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    display_img = frame.copy()
    
    if ball_data:
        x, y, w, h = ball_data['rect']
        cX, cY = ball_data['center']
        area = ball_data['area']
        
        # Draw Window (Green) and Margin (Blue)
        cv.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.rectangle(display_img, (x - margin, y - margin), 
                     (x + w + margin, y + h + margin), (255, 0, 0), 1)
        
        # Draw Centroid and Info
        cv.circle(display_img, (cX, cY), 5, (0, 0, 255), -1)
        text = f"({cX},{cY}) Area:{int(area)}"
        cv.putText(display_img, text, (x, y - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Side-by-side stack
    return np.hstack((display_img, mask_bgr))



def bucketballsmission(start_state=0, target_color="B"):

   ######## FOR DEBUGGING
    debug_dir = "VisionOutput/Debug_mission"
    os.makedirs(debug_dir, exist_ok=True)

    ballmap_dir = "VisionOutput/BallMap"
    os.makedirs(ballmap_dir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ballmap_log = f"{ballmap_dir}/run_{run_id}.csv"
    with open(ballmap_log, 'w') as f:
        f.write("frame,timestamp,color,x_r,y_r,z_r\n")
    print(f"% BallMap log: {ballmap_log}")

    frame_idx = 0
   # ########################
    prev_error_y = 0


    state = start_state 
    print(f"% Mission Started for color: {target_color}")
    t_prev = t.time()
    
    while not service.stop:
        # Grab a fresh frame at the start of the loop
        ok, img, imgTime = cam.getRawFrame()
        if not ok:
            continue # Wait for next frame
        frame_idx += 1 # FOR DEBUGGING

        ###### SETUP IN ORDER TO TRACK FPS
        t_now = t.time()
        loop_time = (t_now - t_prev) * 1000  # conversion to milliseconds
        fps = 1.0 / (t_now - t_prev)
        t_prev = t_now

        print(f"% FPS: {fps:.1f} | Loop: {loop_time:.1f} ms")
        ###################################


        ###### the picamera gives an image in RGB coordinates, but opecv works with BGR by default
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)




        if state == 0: # LOCALIZING (Discrete Search)
            # 1. First, make sure we are still before processing
            # We only "Look" every few frames to ensure the robot has actually stopped
            if frame_idx % 5 == 0: 
                service.send("robobot/cmd/ti", "rc 0 0") # Stop
                t.sleep(0.1) # Brief pause for mechanical stabilization
                
                # Grab a fresh frame AFTER stopping (you might need to flush the buffer)
                ok, img, _ = cam.getRawFrame() 
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                
                found, ball = localize_ball_yolo(img, color=target_color)
                
                if found:
                    print("% Ball found! Locking on.")
                    last_rect = ball['rect']
                    state = 1
                else:
                    print("% Nothing here. Incrementing search...")
                    # 2. Pulse the rotation: Spin at 0.5 speed for 0.2 seconds
                    service.send("robobot/cmd/ti", "rc 0 0.5")
                    t.sleep(0.2) # Adjust this for your ~45 degree goal
                    service.send("robobot/cmd/ti", "rc 0 0") # Stop again
                


        elif state == 1: # ALIGNING
            ball = track_ball_window(img, color=target_color, prev_window=last_rect)
            if ball:

                last_rect = ball['rect']
                cX, cY = ball['center']

                error_x = image_center_x - cX

                if abs(error_x) <= 3:
                    service.send("robobot/cmd/ti", "rc 0 0")
                    print("% Aligned! Driving to ball...")
                    state = 2
                else:
                    # P-Control for turning
                    turn_vel = Kp_turn * error_x
                    # Limit max turn speed for safety
                    turn_vel = max(-0.5, min(0.5, turn_vel))
                    service.send("robobot/cmd/ti", f"rc 0 {turn_vel:.3f}")



                ####### FOR DEBUGGING: every 10 frames save images with information about ball
                ####### NOTE: Recording/saving images decreses FPS, which decreases control performance
                if frame_idx % 10 == 0:
                    cv.imwrite(f"{debug_dir}/{frame_idx:04d}_align.jpg",
                                create_debug_view(img, ball, target_color))
                    _log_detections(img, frame_idx, ballmap_log)
                #############################


            else:
                print("lost Ball")
                state = 0  # go back to localization



        elif state == 2: # TRACKING
            ball = track_ball_window(img, color=target_color, prev_window=last_rect)
            if ball:
                last_rect = ball['rect']
                cX, cY = ball['center']

                error_y = target_y - cY
                error_x = image_center_x - cX

                # --- PD CALCULATION ---
                derivative_y = error_y - prev_error_y

                if abs(error_y) <= 10 and abs(error_x) <= 3:
                    service.send("robobot/cmd/ti", "rc 0 0")
                    print("% MISSION COMPLETE: Ball Reached.")
                    cv.imwrite(f"{debug_dir}/{frame_idx:04d}_final.jpg",
                               create_debug_view(img, ball, target_color))
                    _log_detections(img, frame_idx, ballmap_log)
                    break
                else:
                    # fwd_vel = Kp_fwd * error_y
                    fwd_vel = (Kp_fwd * error_y) + (Kd_fwd * derivative_y)
                    fwd_vel = max(0, min(0.3, fwd_vel))
                    prev_error_y = error_y

                    turn_vel = Kp_turn * error_x
                    turn_vel = max(-0.5, min(0.5, turn_vel))
                    service.send("robobot/cmd/ti", f"rc {fwd_vel:.2f} {turn_vel:.3f}")


                ####### FOR DEBUGGING: every 10 frames save images with information about ball
                ####### NOTE: Recording/saving images decreses FPS, which decreases control performance
                if frame_idx % 10 == 0:
                    cv.imwrite(f"{debug_dir}/{frame_idx:04d}_track.jpg",
                                create_debug_view(img, ball, target_color))
                    _log_detections(img, frame_idx, ballmap_log)
                #############################


            
            else:
                print("lost Ball")
                state = 0  # go back to localization
                prev_error_y = 0





############
############
########### VERY IMPORTANT: the function below creates a video of the entire run
########### this maskes the FPS drop so much that the tracker stop working as intended
########### I use this function just to make a cute little video, but DO NOT use it for real testing
def bucketballsmissionwithRec(start_state=0, target_color="B"):

   ######## FOR DEBUGGING
    debug_dir = "VisionOutput/Debug_mission"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
   # ########################
    prev_error_y = 0


    state = start_state 
    print(f"% Mission Started for color: {target_color}")
    t_prev = t.time()
    
    while not service.stop:
        # Grab a fresh frame at the start of the loop
        ok, img, imgTime = cam.getRawFrame()
        if not ok:
            continue # Wait for next frame

        ###### SETUP IN ORDER TO TRACK FPS
        t_now = t.time()
        loop_time = (t_now - t_prev) * 1000  # conversion to milliseconds
        fps = 1.0 / (t_now - t_prev)
        t_prev = t_now

        print(f"% FPS: {fps:.1f} | Loop: {loop_time:.1f} ms")
        ###################################


        ###### the picamera gives an image in RGB coordinates, but opecv works with BGR by default
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        if state == 0: # LOCALIZING
            found, ball = localize_ball_lowest_contour(img, color=target_color)
            if found:
                print("% Ball found")

                ####### FOR DEBUGGING/VISUALIZATION
                record_frame(create_debug_view(img, ball, target_color), output_dir=debug_dir)
                ########


                last_rect = ball['rect'] # Initialize tracker window
                state = 1
            else:
                print("ball not found")
                break


        elif state == 1: # ALIGNING
            ball = track_ball_window(img, color=target_color, prev_window=last_rect)
            if ball:

                last_rect = ball['rect']
                cX, cY = ball['center']

                error_x = image_center_x - cX

                if abs(error_x) <= 3:
                    service.send("robobot/cmd/ti", "rc 0 0")
                    print("% Aligned! Driving to ball...")
                    state = 2
                else:
                    # P-Control for turning
                    turn_vel = Kp_turn * error_x
                    # Limit max turn speed for safety
                    turn_vel = max(-0.5, min(0.5, turn_vel))
                    service.send("robobot/cmd/ti", f"rc 0 {turn_vel:.3f}")



                ####### FOR DEBUGGING: every 10 frames save images with information about ball                    
                record_frame(create_debug_view(img, ball, target_color), output_dir=debug_dir)
                #############################


            else:
                print("lost Ball")
                state = 0  # go back to localization



        elif state == 2: # TRACKING
            ball = track_ball_window(img, color=target_color, prev_window=last_rect)
            if ball:
                last_rect = ball['rect']
                cX, cY = ball['center']

                error_y = target_y - cY
                error_x = image_center_x - cX

                # --- PD CALCULATION ---
                derivative_y = error_y - prev_error_y

                if abs(error_y) <= 10 and abs(error_x) <= 3:
                    service.send("robobot/cmd/ti", "rc 0 0")
                    print("% MISSION COMPLETE: Ball Reached.")
                    record_frame(create_debug_view(img, ball, target_color), output_dir=debug_dir)
                    break
                else:
                    # fwd_vel = Kp_fwd * error_y
                    fwd_vel = (Kp_fwd * error_y) + (Kd_fwd * derivative_y)
                    fwd_vel = max(0, min(0.3, fwd_vel))
                    prev_error_y = error_y

                    turn_vel = Kp_turn * error_x
                    turn_vel = max(-0.5, min(0.5, turn_vel))
                    service.send("robobot/cmd/ti", f"rc {fwd_vel:.2f} {turn_vel:.3f}")


                ####### FOR DEBUGGING: every 10 frames save images with information about ball
                record_frame(create_debug_view(img, ball, target_color), output_dir=debug_dir)
                #############################


            
            else:
                print("lost Ball")
                state = 0  # go back to localization
                prev_error_y = 0