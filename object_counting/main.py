from ultralytics import YOLO
from tqdm import tqdm 

# Load a pretrained YOLOv8n model
model = YOLO('yolov8x.pt')

# Define path to video file
source = '/users/wylin2/vulcan/object_counting/test.mp4'

import supervision as sv
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from supervision.video.source import get_video_frames_generator

video_info = VideoInfo.from_video_path(source)

generator = get_video_frames_generator(source)



LINE_START = Point(300, 900)
LINE_END = Point(500, 900)
line_counter = LineCounter(start=LINE_START, end=LINE_END)

from supervision.tools.line_counter import LineCounter
from supervision.geometry.dataclasses import Point


with VideoSink(source, video_info) as sink:
   for frame in tqdm(generator, total=video_info.total_frames):
       
        result = model(frame)
        detections = sv.Detections.from_ultralytics(result)
       
        line_counter.update(detections=detections)

        # line_counter.