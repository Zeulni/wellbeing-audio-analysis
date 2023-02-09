import cv2
import sys
import os
import pickle

from src.audio.ASD.model.faceDetector.s3fd import S3FD


class FaceDetector:
    def __init__(self, device, video_path, frames_face_tracking, face_det_scale, pywork_path):
        self.device = device
        self.video_path = video_path
        self.frames_face_tracking = frames_face_tracking
        self.face_det_scale = face_det_scale
        self.pywork_path = pywork_path
        
    def s3fd_face_detection(self):
        # GPU: Face detection, output is the list contains the face location and score in this frame
        DET = S3FD(device=self.device)

        # Instead of using the stored images in Pyframes, load the images from the video (which is stored at videoPath) and go with the detection through each frame
        cap = cv2.VideoCapture(self.video_path)
        
        # TODO: Interpolate linearly between the bounding boxes of the previous and next frame
        # Instead of going through every frame for the face detection, we go through every xth (e.g. 10th) frame and then use the bounding boxes from the previous frame to track the faces in the next frames
        # This is done to reduce the number of frames that need to be processed for the face detection
        dets = []
        fidx = 0
        while(cap.isOpened()):
            ret, image = cap.read()
            if ret == False:
                break
            if fidx%self.frames_face_tracking == 0:
                imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[self.face_det_scale])
                dets.append([])
                for bbox in bboxes:
                    dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
            else:
                dets.append([])
                for bbox in dets[-2]:
                    dets[-1].append({'frame':fidx, 'bbox':bbox['bbox'], 'conf':bbox['conf']})
            sys.stderr.write('%s-%05d; %d dets\r' % (self.video_path, fidx, len(dets[-1])))
            fidx += 1
        

        savePath = os.path.join(self.pywork_path,'faces.pckl')
        with open(savePath, 'wb') as fil:
            pickle.dump(dets, fil)
            
        return dets        