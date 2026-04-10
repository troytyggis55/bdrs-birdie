#!/usr/bin/env python3

#/***************************************************************************
#*   Copyright (C) 2024 by DTU
#*   jcan@dtu.dk
#*
#*
#* The MIT License (MIT)  https://mit-license.org/
#*
#* Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#* and associated documentation files (the “Software”), to deal in the Software without restriction,
#* including without limitation the rights to use, copy, modify, merge, publish, distribute,
#* sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
#* is furnished to do so, subject to the following conditions:
#*
#* The above copyright notice and this permission notice shall be included in all copies
#* or substantial portions of the Software.
#*
#* THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
#* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#* THE SOFTWARE. */

#import sys
#import threading
import time as t
#import select
import numpy as np
import cv2 as cv
from datetime import *
from setproctitle import setproctitle
# robot function
from spose import pose
from sir import ir
from srobot import robot
from scam import cam
from sedge import edge
from sgpio import gpio
from scam import cam
from uservice import service

from CamVision.pictures import *
from CamVision.bucketballsyolo import *
from CamVision.visualcontrol import *
from CamVision.bucketballsmission import *
from pathfinding.simulate_ball_mission import simulate_ball_mission

############################################################

def imageAnalysis(save):
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
      edge.paint(img)
      if not gpio.onPi:
        try:
          cv.imshow('frame for analysis', img)
        except:
          print("% mqtt-client::imageAnalysis: failed to show camera image");
      if save:
        fn = f"image_{imgTime.strftime('%Y_%b_%d_%H%M%S_')}{cam.cnt:03d}.jpg"
        cv.imwrite(fn, img)
        if not service.args.silent:
          print(f"% Saved image {fn}")
      else:
        print("# imageAnalysis:: image not saved")
      pass
    pass
  pass

############################################################

stateTime = datetime.now()

def stateTimePassed():
  return (datetime.now() - stateTime).total_seconds()

############################################################

def driveOneMeter():
  state = 0
  pose.tripBreset()
  print("% Driving 1m -------------------------")
  service.send("robobot/cmd/T0","leds 16 0 100 0") # green
  while not (service.stop):
    if state == 0: # wait for start signal
      service.send("robobot/cmd/ti","rc 0.2 0.0") # (forward m/s, turn-rate rad/sec)
      service.send("robobot/cmd/T0","servo 1 -800 300") # (servo up slow)
      state = 1
    elif state == 1:
      if pose.tripB > 1.0 or pose.tripBtimePassed() > 15:
        service.send("robobot/cmd/ti","rc 0.0 0.0") # (forward m/s, turn-rate rad/sec)
        service.send("robobot/cmd/T0","servo 1 0 0") # (servo front fast)
        state = 2
      pass
    elif state == 2:
      if abs(pose.velocity()) < 0.001:
        state = 99
    else:
      print(f"# drive 1m drove {pose.tripB:.3f}m in {pose.tripBtimePassed():.3f} seconds")
      service.send("robobot/cmd/ti","rc 0.0 0.0") # (forward m/s, turn-rate rad/sec)
      break;
    print(f"# drive {state}, now {pose.tripB:.3f}m in {pose.tripBtimePassed():.3f} seconds; left {edge.posLeft}, right {edge.posRight}")
    t.sleep(0.05)
  pass
  service.send("robobot/cmd/T0","leds 16 0 0 0") # end
  print("% Driving 1m ------------------------- end")

####################################################################3

def driveToLine():
  state = 0
  pose.tripBreset()
  dist_to_line = 0;
  print("% Driving to line ---------------------- right ir start ---")
  service.send("robobot/cmd/T0", "leds 16 0 100 0") # green
  while not (service.stop):
    if state == 0: # forward towards line
      if ir.ir[0] < 0.2:
        service.send("robobot/cmd/ti","rc 0.2 0.0") # (forward m/s, turn-rate rad/sec)
        service.send("robobot/cmd/T0/","lognow 3") # (start Teensy log)
        service.send("robobot/cmd/T0","servo 1 -800 300") # (servo up slow)
        state = 1
    elif state == 1:
      if pose.tripB > 1.0 or pose.tripBtimePassed() > 15:
        service.send("robobot/cmd/ti/","rc 0.0 0.0") # (forward m/s, turn-rate rad/sec)
        state = 2
      if edge.lineValidCnt > 4:
        # start follow line
        edge.lineControl(0.2, True)
        service.send("robobot/cmd/T0","servo 1 0 0") # (move servo to position 0 - front)
        dist_to_line = pose.tripB
        pose.tripBreset()
        print(" to state 10")
        state = 10
      pass
    elif state == 2:
      if abs(pose.velocity()) < 0.001:
        print(" to state 99")
        state = 99
    elif state == 10:
      if edge.lineValidCnt < 2:
        edge.lineControl(0, True)
        service.send("robobot/cmd/ti","rc 0.0 0.0") # (forward m/s, turn-rate rad/sec)
        print(" to state 2")
        pose.tripBreset()
        state = 2
    else:
      print(f"# drive to line {dist_to_line:.3f}m, then along line {pose.tripB:.3f}m in {pose.tripBtimePassed():.3f} seconds")
      service.send("robobot/cmd/ti","rc 0.0 0.0") # (forward m/s, turn-rate rad/sec)
      service.send("robobot/cmd/T0","servo 1 500 200") # (move servo down slow)
      break;
    # print(f"# drive {state}, now {pose.tripB:.3f}m in {pose.tripBtimePassed():.3f} seconds, line valid cnt = {edge.lineValidCnt}")
    t.sleep(0.01)
  pass
  service.send("robobot/cmd/T0","leds 16 0 0 0") # end
  print("% Driving to line ------------------------- end")

####################################################################3

def driveTurnPiX2():
  state = 0
  pose.tripBreset()
  print("% Driving a Pi turn -------------------------")
  service.send("robobot/cmd/T0","leds 16 0 100 0") # green
  while not (service.stop):
    if state == 0: # wait for start signal
      service.send("robobot/cmd/ti","rc 0.0 0.5") # (forward m/s, turn-rate rad/sec)
      state = 1
    elif state == 1:
      if pose.tripBh > 2 * np.pi or pose.tripBtimePassed() > 15:
        service.send("robobot/cmd/ti","rc 0.0 0.0") # (forward m/s, turn-rate rad/sec)
        state = 2
      pass
    elif state == 2:
      if abs(pose.velocity()) < 0.001 and abs(pose.turnrate()) < 0.001:
        state = 99
    else:
      print(f"# drive turned {pose.tripBh:.3f} rad in {pose.tripBtimePassed():.3f} seconds")
      service.send("robobot/cmd/ti","rc 0.0 0.0") # (forward m/s, turn-rate rad/sec)
      break;
    print(f"# turn {state}, now {pose.tripBh:.3f} rad in {pose.tripBtimePassed():.3f} seconds; left {edge.posLeft}, right {edge.posRight}")
    t.sleep(0.05)
  pass
  service.send("robobot/cmd/T0","leds 16 0 0 0") # end
  print("% Driving a Pi turn ------------------------- end")

####################################################################3

def loop():
  from ulog import flog
  state = 0
  images = 0
  ledon = True
  oldstate = -1
  service.send("robobot/cmd/T0", "leds 16 30 30 0") # LED 16: yellow - waiting
  if service.args.meter:
    state = 101 # run 1m
  elif service.args.pi:
    state = 102 # turn 180 deg
  elif service.args.edge:
    state = 103 # find edge and follow line


  elif service.args.photo is not None:
    state = 1600
  elif service.args.yolo:
    state = 1700
  elif service.args.mask:
    state = 1800
  elif service.args.record:
    state = 1900
  elif service.args.bucketballsmission:
    state = 2000
  elif service.args.simulate_ball:
    state = 2100

  elif service.args.usestate > 0:
    state = service.args.usestate
  print(f"% Starting at state {state}")
  # elif not service.args.now:
  #   print("% Ready, press start button")
  # main state machine
  edge.lineControl(0, True) # make sure line control is off (velocity 0)
  while not (service.stop):
    if state == 0: # wait for start signal
      start = True # gpio.start() or service.args.now
      if start:
        print("% Starting")
        service.send("robobot/cmd/T0","leds 16 0 0 30") # blue: running
        service.send("robobot/cmd/ti","rc 0.25 0.0") # (forward m/s, turn-rate rad/sec)
        service.send("robobot/cmd/T0","servo 1 100 300") # (servo down slow)
        state = 12 # until no more line
        pose.tripBreset() # use trip counter/timer B
    elif state == 12: # following line
      if pose.tripB > 0.5 or pose.tripBtimePassed() > 10:
        # start turning
        edge.lineControl(0, True) # stop following line
        pose.tripBreset()
        service.send("robobot/cmd/ti","rc 0.1 0.5") # turn left
        service.send("robobot/cmd/T0","servo 1 -800 1000") # (servo up faster)
        state = 14 # turn left
    elif state == 14: # turning left
      if pose.tripBh > np.pi/2 or pose.tripBtimePassed() > 10:
        state = 20 # finished
        service.send("robobot/cmd/ti","rc 0 0") # stop for images
        service.send("robobot/cmd/T0","servo 1 0 1000") # (servo forward faster)
      # print(f"% --- state {state}, h = {pose.tripBh:.4f}, t={pose.tripBtimePassed():.3f}")
    elif state == 20: # image analysis
      imageAnalysis(images == 2)
      images += 1
      # blink LED
      if ledon:
        service.send("robobot/cmd/T0","leds 16 0 64 0")
        gpio.set_value(20, 1)
      else:
        service.send("robobot/cmd/T0","leds 16 0 30 30")
        gpio.set_value(20, 0)
      ledon = not ledon
      # finished?
      if images >= 10 or (not cam.useCam) or stateTimePassed() > 20:
        images = 0
        state = 99
      pass
    elif state == 101:
      driveOneMeter();
      state = 100
    elif state == 102:
      driveTurnPiX2();
      state = 100
    elif state == 103:
      driveToLine()
      state = 100


  
    elif state == 1600:
      target_folder = service.args.photo
      if oldstate != state:
        print("\n% --- MANUAL PHOTO MODE ACTIVE ---")
        print("% Press 'f' + Enter to take picture")
        print("% Press 'q' + Enter to quit") 
      key = check_keyboard()
      if key == 'f':
        print(f"\n% [TRIGGER] Taking picture and Saving it to {target_folder} ...")
        TakePicture(True, target_folder)
      elif key == 'q':
        print("\n% Exiting Photo Mode")
        state = 100




    # This state was created just to see the YOLO performance time
    elif state == 1700:
      if oldstate != state:
        print("% Initializing Raw Camera...")
        cam.setup_raw()
        oldstate = state
      # 1. Grab the frame
      t_start = t.time() # Start timer
      ok, img, imgTime = cam.getRawFrame()
      if ok:
        # 2. Get the labels, annotate and dislplay on recorder
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        results = get_labels(img)
        annotated_img = annotate_frame(img, results)
        print(f"% YOLO Loop Time: {(t.time() - t_start)*1000:.1f} ms") # See the lag

        key = check_keyboard()
        if key == 'q':
          print("\n% [QUIT] Stopping YOLO recording and exiting state.")
          cv.imwrite("YOLO_last.jpg", annotated_img)
          close_recorder()
          state = 100





    elif state == 1800:
      if oldstate != state:
        print("% Initializing Raw Camera...")
        cam.setup_raw()
        oldstate = state
      # 1. Grab the frame
      t_start = t.time() # Start timer
      ok, img, imgTime = cam.getRawFrame()
      if ok:
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        if service.args.mask is None or service.args.mask == "R":
          masked_img=RedMask(img)
        else:
          masked_img=BlueMask(img)
        record_frame(masked_img, output_dir=f"VisionOutput/MaskingRec/{service.args.mask}")
        print(f"% COLOR MASKING Loop Time: {(t.time() - t_start)*1000:.1f} ms") # See the lag

        key = check_keyboard()
        if key == 'q':
          print("\n% [QUIT] Stopping Color Masking recording and exiting state.")
          close_recorder()
          state = 100




    elif state == 1900:
      if oldstate != state:
        print("% Initializing Raw Camera...")
        cam.setup_raw()
        oldstate = state
      # 1. Grab the frame
      t_start = t.time() # Start timer
      ok, img, imgTime = cam.getRawFrame()
      if ok:
        # 2. Get the labels, annotate and dislplay on recorder
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        record_frame(img, output_dir="VisionOutput/NormalRec")
        print(f"% Recording Time: {(t.time() - t_start)*1000:.1f} ms") # See the lag

        key = check_keyboard()
        if key == 'q':
          print("\n% [QUIT] Stopping recording and exiting state.")
          cv.imwrite("last.jpg", img)
          close_recorder()
          state = 100




    elif state == 2000:
      if oldstate != state:
        print("% Initializing Raw Camera...")
        cam.setup_raw()
        oldstate = state
      bucketballsmission(target_color=service.args.bucketballsmission)
      #bucketballsmissionwithRec(target_color=service.args.bucketballsmission)
      #close_recorder()
      state = 100

    elif state == 2100:
      simulate_ball_mission()
      state = 100







    
    else: # abort
      print(f"% Mission finished/aborted; state={state}")
      break
    # allow openCV to handle imshow (if in use)
    # images are almost useless while turning, but
    # used here to illustrate some image processing (painting)
    # if cam.useCam:
    #   imageAnalysis(True)
    #   if not gpio.onPi:
    #     # do not wait is no image is shown
    #     key = cv.waitKey(100) # ms
    #     if key > 0: # e.g. Esc (key=27) pressed with focus on image
    #       break
    #
    # note state change and reset state timer
    if state != oldstate:
      # flog.write(state)
      flog.writeRemark(f"% State change from {oldstate} to {state}")
      print(f"% State change from {oldstate} to {state}")
      oldstate = state
      stateTime = datetime.now()
    # do not loop too fast
    t.sleep(0.1)
    pass # end of while loop
  # end of mission, turn LEDs off and stop
  service.send("robobot/cmd/T0","leds 16 0 0 0")
  gpio.set_value(20, 0)
  edge.lineControl(0, True) # stop following line
  service.send("robobot/cmd/ti","rc 0 0")
  service.send("robobot/cmd/T0","servo 1 0 0")
  t.sleep(0.05)
  pass

############################################################

if __name__ == "__main__":
    if service.process_running("mqtt-client"):
      print("% mqtt-client is already running - terminating")
      print("%   if it is partially crashed in the background, then try:")
      print("%     pkill mqtt-client")
      print("%   or, if that fails use the most brutal kill")
      print("%     pkill -9 mqtt-client")
    else:
      # set title of process, so that it is not just called Python
      setproctitle("mqtt-client")
      print("% Starting")
      # where is the MQTT data server:
      service.setup('localhost') # localhost
      #service.setup('10.197.217.81') # Juniper
      #service.setup('10.197.217.80') # Newton
      # service.setup('bode.local') # Bode
      if service.connected:
        loop()
      service.terminate()
    print("% Main Terminated")
