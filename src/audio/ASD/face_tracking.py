import numpy

from scipy.interpolate import interp1d

class FaceTracker:
    def __init__(self, num_failed_det, min_track, min_face_size) -> None:
        self.num_failed_det = num_failed_det
        self.min_track = min_track
        self.min_face_size = min_face_size
        

      
    # CPU: Face tracking  
    def track_shot_face_tracker(self, faces) -> list:
        iouThres  = 0.5     # Minimum IOU between consecutive face detections
        tracks    = []
        while True:
            track     = []
            for frameFaces in faces:
                for face in frameFaces:
                    if track == []:
                        track.append(face)
                        frameFaces.remove(face)
                    elif face['frame'] - track[-1]['frame'] <= self.num_failed_det:
                        iou = self._FaceTracker__bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        if iou > iouThres:
                            track.append(face)
                            frameFaces.remove(face)
                            continue
                    else:
                        break
            if track == []:
                break
            elif len(track) > self.min_track:
                frameNum    = numpy.array([ f['frame'] for f in track ])
                bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
                frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
                bboxesI    = []
                for ij in range(0,4):
                    interpfn  = interp1d(frameNum, bboxes[:,ij])
                    bboxesI.append(interpfn(frameI))
                bboxesI  = numpy.stack(bboxesI, axis=1)
                if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > self.min_face_size:
                    tracks.append({'frame':frameI,'bbox':bboxesI})
        return tracks
    
    def __bb_intersection_over_union(self, boxA, boxB, evalCol = False) -> float:
        # CPU: IOU Function to calculate overlap between two image
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if evalCol == True:
            iou = interArea / float(boxAArea)
        else:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou