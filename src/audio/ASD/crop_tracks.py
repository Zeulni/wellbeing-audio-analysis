import cv2
import numpy
from scipy import signal
import torchvision.transforms as transforms
import torch

from src.audio.ASD.utils.asd_pipeline_tools import write_to_terminal

class CropTracks:
    def __init__(self, video_path, total_frames, frames_face_tracking, crop_scale):
        self.video_path = video_path
        self.total_frames = total_frames
        self.frames_face_tracking = frames_face_tracking
        self.crop_size = crop_scale
    
    def crop_tracks_from_videos_parallel(self, tracks, file_path_frames_storage) -> tuple:

        # Go through all the tracks and get the dets values (not through the frames yet, only dets)
        # Save for each track the dats in a list
        dets = []
        for track in tracks:
            dets.append({'x':[], 'y':[], 's':[]})
            for fidx, det in enumerate(track['bbox']):
                if fidx%self.frames_face_tracking  == 0:
                    dets[-1]['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
                    dets[-1]['y'].append((det[1]+det[3])/2)
                    dets[-1]['x'].append((det[0]+det[2])/2)
                else:
                    dets[-1]['s'].append(dets[-1]['s'][-1])
                    dets[-1]['y'].append(dets[-1]['y'][-1])
                    dets[-1]['x'].append(dets[-1]['x'][-1])
        
        # Go through all the tracks and smooth the dets values
        for track in dets:	
            track['s'] = signal.medfilt(track['s'], kernel_size=13)
            track['x'] = signal.medfilt(track['x'], kernel_size=13)
            track['y'] = signal.medfilt(track['y'], kernel_size=13)
        
        # Open the video
        vIn = cv2.VideoCapture(self.video_path)
        num_frames = self.total_frames
        cs = self.crop_size

        # Define transformation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255),
            transforms.Lambda(lambda x: x.type(torch.uint8))
        ])
                
        # To not store all the frames in memory, we read the video in chunks and store it to harddrive
        output_file = file_path_frames_storage
        # chunk_size = self.crop_chunk_size  # number of frames to process at a time (and storing into RAM)
            
        length_frames = int(num_frames / self.frames_face_tracking)
        all_faces = numpy.memmap(output_file, mode="w+", shape=(len(tracks), length_frames, 112, 112), dtype=numpy.uint8)
        
        # for chunk_start in range(0, num_frames, chunk_size):
        #     chunk_end = min(chunk_start + chunk_size, num_frames) 
        #     # Loop over every frame, read the frame, then loop over all the tracks per frame and if available, crop the face
            
        for fidx in range(0, num_frames, self.frames_face_tracking):
            vIn.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, image = vIn.read()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for tidx, track in enumerate(tracks):
                # In the current frame, first check whether the track has a bbox for this frame (if yes, perform opererations)
                if fidx in track['frame']:
                    
                    # Get the index of the frame in the track
                    index = numpy.where(track['frame'] == fidx)
                    index = int(index[0][0])
        
                    # Calculate the bsi and pad the image
                    bsi = int(dets[tidx]['s'][index] * (1 + 2 * cs))
                    frame_image = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))

                    bs  = dets[tidx]['s'][index]
                    my  = dets[tidx]['y'][index] + bsi
                    mx  = dets[tidx]['x'][index] + bsi

                    # Crop the face from the image (depending on the track choose the image)
                    face = frame_image[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

                    # Apply the transformations
                    face = transform(face)
                    
                    # Directly write to the chunk_faces array
                    if fidx//self.frames_face_tracking < length_frames:
                        all_faces[tidx, fidx//self.frames_face_tracking, :, :] = face[0, :, :]
        
        # Close the video
        vIn.release()
        
        # Flush the data to disk and close the file
        all_faces.flush()
        del all_faces

        # Create a list where each element has the format {'track':track, 'proc_track':dets}
        proc_tracks = []
        for i in range(len(tracks)):
            proc_tracks.append({'track':tracks[i], 'proc_track':dets[i]})
            
        write_to_terminal("Finished cropping the faces from the videos -- saved them to: " + output_file + "")

        return proc_tracks