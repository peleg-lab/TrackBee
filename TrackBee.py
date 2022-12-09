# install dependencies: opencv-python, numpy
import cv2
import numpy as np
import time
import math
import os

# Print out run conditions:
print(os.path.basename(__file__))
from datetime import datetime
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

def main():
    #
    # USER PARAMETERS:
    #
    results_type = "contour"  # Options: raw, mask, contour, box, track
    input_video_path = "/home/nbonnie/Research/TrackBee/input.mp4"
    write_ouput_to_file = False
    ouput_video_fps = 10
    ouput_video_path = "/home/nbonnie/Research/TrackBee/contour.mp4"
    step_through_frames_with_key = False  # If true, allows the result video to be played 1 frame at a time with every keyboard input

    # Import in mp4 video of chronological bee cluster data
    v = cv2.VideoCapture(input_video_path)
    tracker = EuclideanDistTracker()  # sets up our tracking class

    # Calculate useful video metrics
    width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))  # 910
    height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 640
    frames_count = int(v.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set color threshold:
    # We work in RGB so we can overlay colored contours later
    # The exact low RGB bound equates to a specific temperature of bee around 34.5 C
    # This essentially helps us focus on the important parts of each frame
    lower_color_bound = np.array([151,151,151])
    upper_color_bound = np.array([255,255,255])

    # Start timer
    start_time = time.time()
    video_out = []
    for i in range(frames_count):
        _, frame = v.read(i)  # Grab current frame
        if results_type == "raw":
            video_out.append(frame)
            continue

        # Crop region of interest
        region = frame[0:615, 0:820]  # crop out temperature bar and bottom of frame data

        # Color threshold detection
        mask = cv2.inRange(region, lower_color_bound, upper_color_bound)
        _, mask = cv2.threshold(mask, 151, 255, cv2.THRESH_BINARY) # convert mask into binary
        if results_type == "mask":
            video_out.append(mask)
            continue

        # Find every contour (outline of shapes) in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cont in contours: # Loop through every contour in the current frame
            area = cv2.contourArea(cont) # Calculate single contour area
            if area > 5:  # If the shape is "big enough" (prevents noise)
                if results_type == "contour":
                    cv2.drawContours(frame, [cont], -1, (0,255,0),2)
                if results_type == "box":
                    x, y, w, h = cv2.boundingRect(cont)
                    cv2.rectangle(frame, (x, y), (x+h, y+h), (0,255,0), 2)
                if results_type == "track":
                    x, y, w, h = cv2.boundingRect(cont)
                    detections.append([x,y,w,h])
        if results_type == "contour" or results_type == "box":
            video_out.append(frame)

        if results_type == "track":
            # Tracking by euclidean distance
            box_ids = tracker.update(detections) # Update the tracker with the lastest array of boxes
            for box_id in box_ids: # For every boxid returned from the tracker
                x, y, w, h, bid = box_id
                cv2.putText(frame, str(bid), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)  # Plot ids
                cv2.rectangle(frame, (x, y), (x+h, y+h), (0,255,0), 2)  # Plot boxes
            video_out.append(frame)

    print("--- %s seconds ---" % (time.time() - start_time))

    # Write to file
    size = video_out[0].shape # Grab a frame, extract dimensions
    if write_ouput_to_file: # IF user requested to write results to file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(ouput_video_path, fourcc, ouput_video_fps, (size[1], size[0]))
        for j in range(len(video_out)):
            out.write(video_out[j])

    # Prepare cv2 live figure
    cv2.startWindowThread()
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", int(width), int(height))

    # DISPLAY IMAGES
    for i in range(len(video_out)):
        # time.sleep(.05)  # Uncomment this line for slower playback
        try:
            cv2.imshow("Result", video_out[i])
        except:
            break

        if step_through_frames_with_key:
            key = cv2.waitKey(0)
            if key == 27:   # IF ESCAPE is pressed, exit the video
                print("Escape key pressed, video terminating")
                break
        else:
            key = cv2.waitKey(40)
            if key >= 0:   # IF ANY key is pressed, exit the video
                print("Key press detected, video terminating")
                break

    # Close the window and release device resources
    cv2.destroyAllWindows()
    v.release()

# Tracking algorithm based off nearest distance
class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = x + (w//2)
            cy = y + (h//2)

            # Find out if that object was detected already
            same_object_detected = False
            for bid, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[bid] = (cx, cy)
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, bid])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

if __name__ == '__main__':
    main()
