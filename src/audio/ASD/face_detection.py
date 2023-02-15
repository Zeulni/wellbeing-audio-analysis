import cv2
import sys
import numpy as np

from src.audio.ASD.model.faceDetector.s3fd import S3FD


class FaceDetector:
    def __init__(self, device, video_path, frames_face_tracking, face_det_scale, pywork_path, num_frames) -> None:
        self.device = device
        self.video_path = video_path
        self.frames_face_tracking = frames_face_tracking
        self.face_det_scale = face_det_scale
        self.pywork_path = pywork_path
        self.num_frames = num_frames
        
    def s3fd_face_detection(self):
        # GPU: Face detection, output is the list contains the face location and score in this frame
        DET = S3FD(device=self.device)
        
        # TODO: Interpolate linearly between the bounding boxes of the previous and next frame
        # Instead of going through every frame for the face detection, we go through every xth (e.g. 10th) frame and then use the bounding boxes from the previous frame to track the faces in the next frames
        # This is done to reduce the number of frames that need to be processed for the face detection
        
        # Instead of appending the dets list, preallocate the dets list and then insert the values to save time
        dets = [None] * self.num_frames
        cap = cv2.VideoCapture(self.video_path)
        fidx = 0
        while(cap.isOpened()):
            ret, image = cap.read()
            if ret == False:
                break    

            if fidx%self.frames_face_tracking == 0:
                imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[self.face_det_scale])
                dets[fidx] = []
                for bbox in bboxes:
                    dets[fidx].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
            else:
                dets[fidx] = []
                for bbox in dets[fidx-1]:
                    dets[fidx].append({'frame':fidx, 'bbox':bbox['bbox'], 'conf':bbox['conf']})
                    
            if fidx%100 == 0:  
                sys.stderr.write('%s-%05d; %d dets\r' % (self.video_path, fidx, len(dets[fidx])))
            fidx += 1
            
        cap.release()   
        
        return dets       