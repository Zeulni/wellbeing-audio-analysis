import cv2
import sys
import numpy as np
from scipy.interpolate import interp1d

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
        
        dets= [None] * self.num_frames
        cap_old = cv2.VideoCapture(self.video_path)       
        
        fidx = 0 
        
        while(cap_old.isOpened()):
            ret, image = cap_old.read()
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
                    
            if fidx%100 == 0:  
                sys.stderr.write('%s-%05d; %d dets\r' % (self.video_path, fidx, len(dets[fidx])))
            
            fidx += 1
            
        # Remove all the entrys in the dets list that are 1
        dets = [x for x in dets if x != []]
        
        # Interpolate the face location for the frames without face detection
        # interpolated_dets = self.interpolate_face_location(dets)
            
        cap_old.release()   
        
        return dets       
    
    # def interpolate_face_location(self, dets) -> list:
    #     for i in range(len(dets)):
    #         if dets[i] == []:
    #             # Find nearest frames with bounding boxes
    #             j = i-1
    #             while j >= 0 and dets[j] == []:
    #                 j -= 1
    #             k = i+1
    #             while k < len(dets) and dets[k] == []:
    #                 k += 1
                
    #             if j < 0 or k >= len(dets):
    #                 # Cannot interpolate missing value at beginning or end
    #                 continue
                
    #             # Linearly interpolate bounding boxes
    #             bbox_j = np.array([f['bbox'] for f in dets[j]])
    #             bbox_k = np.array([f['bbox'] for f in dets[k]])
    #             bbox_interp = interp1d([j,k], np.stack([bbox_j, bbox_k]), axis=0)([i])[0]
                
    #             # Create fake detections for missing frame
    #             dets[i] = [{'frame':i, 'bbox':bbox.tolist(), 'conf':1.0} for bbox in bbox_interp]
                
    #     return dets