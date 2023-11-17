import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from numpy.linalg import norm as l2norm
from skimage import transform as trans
import subprocess
from math import floor, ceil
import bisect

import torch
from torchvision import transforms
import json
import math



# from itertools import combinations

lock=threading.Lock()

class VideoManager():  
    def __init__( self ):
        # Model related
        self.swapper_model = []             # insightface swapper model
        self.faceapp_model = []             # insight faceapp model
        self.input_names = []               # names of the inswapper.onnx inputs
        self.input_size = []                # size of the inswapper.onnx inputs
        self.emap = []                      # comes from loading the inswapper model. not sure of data
        self.output_names = []              # names of the inswapper.onnx outputs    
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)                 
        self.GFPGAN_model = []
        self.occluder_model = []
        self.face_parsing_model = []
        self.face_parsing_tensor = []   
        self.codeformer_model = []
        
        #Video related
        self.capture = []                   # cv2 video
        self.is_video_loaded = False        # flag for video loaded state    
        self.video_frame_total = None       # length of currently loaded video
        self.play = False                   # flag for the play button toggle
        self.current_frame = 0              # the current frame of the video
        self.create_video = False
        self.output_video = []       
        self.file_name = []       
        self.vid_qual = []
        
        # Play related
        # self.set_read_threads = []          # Name of threaded function
        self.frame_timer = 0.0      # used to set the framerate during playing
        
        # Queues
        self.action_q = []                  # queue for sending to the coordinator
        self.frame_q = []                   # queue for frames that are ready for coordinator

        self.r_frame_q = []                 # queue for frames that are requested by the GUI
        self.read_video_frame_q = []
        
        # swapping related
        self.source_embedding = []          # array with indexed source embeddings
        self.swap = False                   # flag for the swap enabled toggle
        self.target_facess = []   # array that maps the found faces to source faces    

        self.parameters = []

        self.num_threads = 0
        self.target_video = []

        self.fps = 1.0
        self.temp_file = []

        self.i_image = []
        self.io_binding = False
        self.video_read_success = False
        self.clip_session = []

        self.start_time = []
        self.record = False
        self.output = []
        self.image = []

        self.saved_video_path = []
        self.sp = []
        self.timer = []
        self.fps_average = []
        self.total_thread_time = 0.0
        
        self.start_play_time = []
        self.start_play_frame = []
        
        self.rec_thread = []
        self.markers = []
        self.is_image_loaded = False
        
        self.process_q =    {
                            "Thread":                   [],
                            "FrameNumber":              [],
                            "ProcessedFrame":           [],
                            "Status":                   'clear',
                            "ThreadTime":               []
                            }   
        self.process_qs = []
        self.rec_q =    {
                            "Thread":                   [],
                            "FrameNumber":              [],
                            "Status":                   'clear'
                            }   
        self.rec_qs = []

        self.clip_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Resize((352, 352))])
        
        self.arcface_dst_max = []
        self.arcface_dst_max.append( math.sqrt(( self.arcface_dst[0][0]- self.arcface_dst[1][0])*( self.arcface_dst[0][0]- self.arcface_dst[1][0]) + ( self.arcface_dst[0][1]- self.arcface_dst[1][1])*( self.arcface_dst[0][1]- self.arcface_dst[1][1])) )
        self.arcface_dst_max.append( math.sqrt(( self.arcface_dst[1][0]- self.arcface_dst[4][0])*( self.arcface_dst[1][0]- self.arcface_dst[4][0]) + ( self.arcface_dst[1][1]- self.arcface_dst[4][1])*( self.arcface_dst[1][1]- self.arcface_dst[4][1])) )
        self.arcface_dst_max.append( math.sqrt(( self.arcface_dst[3][0]- self.arcface_dst[4][0])*( self.arcface_dst[3][0]- self.arcface_dst[4][0]) + ( self.arcface_dst[3][1]- self.arcface_dst[4][1])*( self.arcface_dst[3][1]- self.arcface_dst[4][1])) )
        self.arcface_dst_max.append( math.sqrt(( self.arcface_dst[0][0]- self.arcface_dst[3][0])*( self.arcface_dst[0][0]- self.arcface_dst[3][0]) + ( self.arcface_dst[0][1]- self.arcface_dst[3][1])*( self.arcface_dst[0][1]- self.arcface_dst[3][1])) )
        self.arcface_dst_max.append( math.sqrt(( self.arcface_dst[0][0]- self.arcface_dst[4][0])*( self.arcface_dst[0][0]- self.arcface_dst[4][0]) + ( self.arcface_dst[0][1]- self.arcface_dst[4][1])*( self.arcface_dst[0][1]- self.arcface_dst[4][1])) )
        self.arcface_dst_max.append( math.sqrt(( self.arcface_dst[1][0]- self.arcface_dst[3][0])*( self.arcface_dst[1][0]- self.arcface_dst[3][0]) + ( self.arcface_dst[1][1]- self.arcface_dst[3][1])*( self.arcface_dst[1][1]- self.arcface_dst[3][1])) )        



    def load_target_video( self, file ):
        # If we already have a video loaded, release it
        if self.capture:
            self.capture.release()
            
        # Open file                
        self.capture = cv2.VideoCapture(file)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        if not self.capture.isOpened():
            print("Cannot open file: ", file)
            
        else:
            self.target_video = file
            self.is_video_loaded = True
            self.is_image_loaded = False
            self.video_frame_total = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.play = False 
            self.current_frame = 0
            self.frame_timer = time.time()
            self.frame_q = []            
            self.r_frame_q = []             
            self.target_facess = []
            self.add_action("set_slider_length",self.video_frame_total-1)

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)        
        success, image = self.capture.read() 
        
        if success:
            crop = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
            temp = [crop, False]
            self.frame_q.append(temp)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
    
    def load_target_image(self, file):
        self.is_video_loaded = False
        self.play = False 
        self.frame_q = []            
        self.r_frame_q = [] 
        self.target_facess = []
        self.image = cv2.imread(file) # BGR
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) # RGB
        temp = [self.image, False]
        self.frame_q.append(temp)

        self.is_image_loaded = True
        
    def load_null(self):
        self.is_video_loaded = False
        self.is_image_loaded = False
        self.play = False 
        self.frame_q = []            
        self.r_frame_q = [] 
        self.target_facess = []
    
    ## Action queue
    def add_action(self, action, param):
        temp = [action, param]
        self.action_q.append(temp)    
    
    def get_action_length(self):
        return len(self.action_q)

    def get_action(self):
        action = self.action_q[0]
        self.action_q.pop(0)
        return action
     
    ## Queues for the Coordinator
    def get_frame(self):
        frame = self.frame_q[0]
        self.frame_q.pop(0)
        return frame
    
    def get_frame_length(self):
        return len(self.frame_q)  
        
    def get_requested_frame(self):
        frame = self.r_frame_q[0]
        self.r_frame_q.pop(0)
        return frame
    
    def get_requested_frame_length(self):
        return len(self.r_frame_q)          
    

    def get_requested_video_frame(self, frame):  
        if self.is_video_loaded == True:
        
            if self.play == True:            
                self.play_video("stop")
                self.process_qs = []
                
            self.current_frame = int(frame)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            success, target_image = self.capture.read() #BGR
            # self.current_frame += 1
            if success:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB) #RGB 
                if not self.swap:   
                    temp = [target_image, self.current_frame] #temp = RGB
                else:  
                    temp = [self.swap_video(target_image, self.current_frame, False), self.current_frame] # temp = RGB

                self.r_frame_q.append(temp)  
    
    # Here we want to make adjustments to the parameters without the swap reading from existing markers
    def get_requested_video_frame_parameters(self, frame):  
        temp = []
        if self.is_video_loaded == True:
        
            # if self.play == True:            
                # self.play_video("stop")
                # self.process_qs = []
                
            self.current_frame = int(frame)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            success, target_image = self.capture.read() #BGR
            # self.current_frame += 1
            if success:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB) #RGB 
                if not self.swap:   
                    temp = [target_image, self.current_frame] #temp = RGB
                else:  
                    temp = [self.swap_video(target_image, self.current_frame, True), self.current_frame] # temp = RGB

                self.r_frame_q.append(temp)  
        elif self.is_image_loaded == True:
            if not self.swap:   
                temp = [self.image, self.current_frame] # image = RGB
        
            else:  

                temp2 = self.swap_video(self.image, self.current_frame, True)
                temp = [temp2, self.current_frame] # image = RGB
            
            self.r_frame_q.append(temp)  

    def play_video(self, command):        
        if command == "play":
            self.play = True
            self.fps_average = []            
            self.process_qs = []
            self.current_frame +=1
            for i in range(self.num_threads):
                    new_process_q = self.process_q.copy()
                    self.process_qs.append(new_process_q)
            # # Attempt at frame_sync
            # self.start_play_time = time.time()
            # self.start_play_frame = self.current_frame
            
        if command == "stop":
            self.play = False
            self.add_action("stop_play", True)
            

        if command == "record":
            self.record = True
            self.play = True
            self.total_thread_time = 0.0

            self.process_qs = []
            
            for i in range(self.num_threads):
                    new_process_q = self.process_q.copy()
                    self.process_qs.append(new_process_q)

           # Initialize
            self.timer = time.time()
            self.current_frame +=1
            frame_width = int(self.capture.get(3))
            frame_height = int(self.capture.get(4))

            self.start_time = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))            
            
            self.file_name = os.path.splitext(os.path.basename(self.target_video))
            base_filename =  self.file_name[0]+"_"+str(time.time())[:10]
            self.output = os.path.join(self.saved_video_path, base_filename)
            self.temp_file = self.output+"_temp"+self.file_name[1]  

            # self.vid_obj = cv2.VideoWriter(self.temp_file, cv2.VideoWriter_fourcc(*'mpv4'), self.fps, (frame_width, frame_height))

            args =  ["ffmpeg", 
                    '-hide_banner',
                    '-loglevel',    'error',
                    "-an",       
                    "-r",           str(self.fps),
                    "-i",           "pipe:",
                    # '-g',           '25',
                    "-vf",          "format=yuvj420p",
                    "-c:v",         "libx264",
                    "-crf",         str(self.vid_qual),
                    "-r",           str(self.fps),
                    "-s",           str(frame_width)+"x"+str(frame_height),
                    self.temp_file]  
            
            self.sp = subprocess.Popen(args, stdin=subprocess.PIPE)

         
                
    # @profile
    def process(self):
        process_qs_len = range(len(self.process_qs))

        # Add threads to Queue
        if self.play == True and self.is_video_loaded == True:
            for item in self.process_qs:
                if item['Status'] == 'clear' and self.current_frame < self.video_frame_total:
                    item['Thread'] = threading.Thread(target=self.thread_video_read, args = [self.current_frame]).start()
                    item['FrameNumber'] = self.current_frame
                    item['Status'] = 'started'
                    item['ThreadTime'] = time.time()

                    self.current_frame += 1
                    break
          
        else:
            self.play = False

        # Always be emptying the queues
        time_diff = time.time() - self.frame_timer

        if not self.record and time_diff >= 1.0/float(self.fps) and self.play:
            # Find process-qs with lowest(next) FrameNumber
            try:
                mfn = min(self.process_qs, key=lambda x: x['FrameNumber'] if x['FrameNumber'] != [] else 999999999)
            except (TypeError):
                print('did not find minimum frame')

            if mfn['Status'] == 'finished':
                temp = [mfn['ProcessedFrame'], mfn['FrameNumber']]
                self.frame_q.append(temp)

                # Report fps, other data
                self.fps_average.append(1.0/time_diff)
                if len(self.fps_average) >= floor(self.fps):
                    fps = round(np.average(self.fps_average), 2)
                    msg = "%s fps, %s process time" % (fps, round(mfn['ThreadTime'], 4))
                    self.add_action("send_msg", msg)
                    self.fps_average = []

                mfn['Status'] = 'clear'
                mfn['Thread'] = []
                mfn['FrameNumber'] = []
                mfn['ThreadTime'] = []
                self.frame_timer = time.time()
                
                # # Attempt at frame sync
                # frame_time = 1.0/float(self.fps)
                # expected_frame = int((time.time()-self.start_play_time)*self.fps) + self.start_play_frame
                # delta_frames = floor(expected_frame-self.current_frame)
                # if delta_frames > 0:
                    # self.current_frame = expected_frame  
                    
        elif self.record:
            empty_count = 0
            empty_count_rec = 0
            # try:
                # # return process_qs item with minimum frame number
                # mfn = min(self.process_qs, key=lambda x: x['FrameNumber'] if x['FrameNumber'] != [] else 999999999)
            # except (TypeError):
                # print('line 371 error')
             
            temp = []
            for process in self.process_qs:
                if process['FrameNumber'] != []:
                    temp.append( process['FrameNumber'])
                else:
                    temp.append( 9999999999)
                
            # print(temp)
            index_min = np.argmin(temp)
            
            mfn = self.process_qs[index_min]
            # print(mfn)
            
            # If the swapper thread has finished generating a frame
            if mfn['Status'] == 'finished':

                image = mfn['ProcessedFrame']  
                # self.vid_obj.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                pil_image = Image.fromarray(image)
                pil_image.save(self.sp.stdin, 'JPEG')   

                temp = [image, mfn['FrameNumber']]
                self.frame_q.append(temp)

                self.total_thread_time += mfn['ThreadTime']
                mfn['Status'] = 'clear'
                mfn['FrameNumber'] = []
                mfn['Thread'] = []
                self.frame_timer = time.time()


            empty_count = 0
            for process in self.process_qs:
                if process['Status'] == 'clear':
                    empty_count +=1
                    
            # Close video and process
            if empty_count == self.num_threads:

                stop_time = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))
                if stop_time == 0:
                    stop_time = float(self.video_frame_total) / float(self.fps)
                
                self.sp.stdin.close()
                self.sp.wait()
                # self.vid_obj.release()
                self.play_video("stop")

                orig_file = self.target_video
                final_file = self.output+self.file_name[1]
                self.add_action("send_msg", "adding audio...")    
                args = ["ffmpeg",
                        '-hide_banner',
                        '-loglevel',    'error',
                        "-i", self.temp_file,
                        "-ss", str(self.start_time), "-to", str(stop_time), "-i",  orig_file,
                        "-c",  "copy", # may be c:v
                        "-map", "0:v:0", "-map", "1:a:0?",
                        "-shortest",
                        final_file]
                
                four = subprocess.run(args)

                
                os.remove(self.temp_file)

                timef= time.time() - self.timer 
                t_time = self.total_thread_time/self.video_frame_total
                self.record = False
                msg = "Total time: %s s." % (round(timef,1))
                print(msg)
                self.add_action("send_msg", msg) 
    # @profile
    def thread_video_read(self, frame_number):  
        with lock:
            success, target_image = self.capture.read()

        if success:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            if not self.swap:
                temp = [target_image, frame_number]
            
            else:
                temp = [self.swap_video(target_image, frame_number, False), frame_number]
            
            for item in self.process_qs:
                if item['FrameNumber'] == frame_number:
                    item['ProcessedFrame'] = temp[0]
                    item['Status'] = 'finished'
                    item['ThreadTime'] = time.time() - item['ThreadTime']
                    break

    def load_source_embeddings(self, source_embeddings):
        self.source_embedding = []
        for i in range(len(source_embeddings)):
            self.source_embedding.append(source_embeddings[i]["Embedding"])

    def set_swapper_model(self, swapper, emap):
        self.swapper_model = swapper
        self.emap = emap
        
        # Get in/out size and create some data
        inputs =  self.swapper_model.get_inputs()
        for inp in inputs:
            self.input_names.append(inp.name)
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_size = tuple(input_shape[2:4][::-1])
        
        outputs = self.swapper_model.get_outputs()
        for out in outputs:
            self.output_names.append(out.name)

    # def set_faceapp_model(self, faceapp):
        # self.faceapp_model = faceapp
    
    # @profile
    def swap_video(self, target_image, frame_number, change_parameters):   
        parameters = self.parameters.copy()

        if self.markers and not change_parameters:
            temp=[]
            for i in range(len(self.markers)):
                temp.append(self.markers[i]['frame'])
            idx = bisect.bisect(temp, frame_number)
            
            parameters = self.markers[idx-1]['parameters'].copy()
        
        # Find faces, returns all faces
        orientation = int(parameters['OrientationAmount'][0]/90)

        for i in range(orientation):
            target_image = cv2.rotate(target_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        found_faces = self.faceapp_model.get(target_image, max_num=10)
        
        if found_faces:
            img = target_image.copy() # img = RGB
            
            # Loop through target faces to see if they match our target embeddings
            for fface in found_faces:
                for tface in self.target_facess:
                    # sim between face in video and already found face
                    sim = self.findCosineDistance(fface.embedding, tface["Embedding"])

                    # if the face[i] in the frame matches afound face[j] AND the found face is active (not []) 
                    threshhold = parameters["ThresholdAmount"][0]/100.0
                    if parameters["ThresholdState"]:
                        threshhold = 2.0

                    if sim<float(threshhold) and tface["SourceFaceAssignments"]:
                        s_e =  tface["AssignedEmbedding"]
                        img = self.swap_core(img, fface.kps, s_e, orientation, parameters, fface.bbox) 
                    else:
                        for i in range(orientation):
                            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
        else:
            for i in range(orientation):
                target_image = cv2.rotate(target_image, cv2.ROTATE_90_CLOCKWISE)
                
            
            return target_image

    
            
       
    def findCosineDistance(self, vector1, vector2):

        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

    def CosineSimilarity(self, test_vec, source_vecs):

        cos_dist = 0
        for source_vec in source_vecs:
            cos_dist += self.findCosineDistance(test_vec, source_vec)
        return cos_dist/len(source_vecs)

    # @profile    
    def swap_core(self, img, kps, s_e, rot, parameters, bbox): # img = RGB

        # 512 transforms
        ratio = 4.0
        diff_x = 8.0*ratio
        dst = self.arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst)
        M512 = tform.params[0:2, :]
        IM512 = cv2.invertAffineTransform(M512)
        
        # orig_bbox = cv2.transform(np.array([[[0,0], [0,512], [512,0], [512,512]]]), np.array(IM512))

        # option 2
        kps_dist = []
        kps_dist.append( math.sqrt((kps[0][0]-kps[1][0])*(kps[0][0]-kps[1][0]) + (kps[0][1]-kps[1][1])*(kps[0][1]-kps[1][1])) )
        kps_dist.append( math.sqrt((kps[1][0]-kps[4][0])*(kps[1][0]-kps[4][0]) + (kps[1][1]-kps[4][1])*(kps[1][1]-kps[4][1])) )
        kps_dist.append( math.sqrt((kps[3][0]-kps[4][0])*(kps[3][0]-kps[4][0]) + (kps[3][1]-kps[4][1])*(kps[3][1]-kps[4][1])) )
        kps_dist.append( math.sqrt((kps[0][0]-kps[3][0])*(kps[0][0]-kps[3][0]) + (kps[0][1]-kps[3][1])*(kps[0][1]-kps[3][1])) )
        kps_dist.append( math.sqrt((kps[0][0]-kps[4][0])*(kps[0][0]-kps[4][0]) + (kps[0][1]-kps[4][1])*(kps[0][1]-kps[4][1])) )
        kps_dist.append( math.sqrt((kps[1][0]-kps[3][0])*(kps[1][0]-kps[3][0]) + (kps[1][1]-kps[3][1])*(kps[1][1]-kps[3][1])) )
        
        # max distance index between all facial features in frame size
        kps_dist_max_index = kps_dist.index(max(kps_dist))   
        kps_dist_max = kps_dist[kps_dist_max_index]
        
        # distance between same features from arcface reference
        arcface_distance_max = self.arcface_dst_max[kps_dist_max_index]
        kps_ratio = kps_dist_max / arcface_distance_max
        # option 2

        original_face_512 = cv2.warpAffine(img, M512, (512,512), borderValue=0.0)
        original_face_256 = cv2.resize(original_face_512, (256,256))
        original_face = cv2.resize(original_face_256, (128, 128))        
        
        #Normalize source embedding
        n_e = s_e / l2norm(s_e)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        
        swapped_face = original_face.copy()
        previous_face = []
        
        itex = int(parameters['StrengthAmount'][0]/100)+1
        for i in range(itex):
            previous_face = swapped_face
            blob = cv2.dnn.blobFromImage(swapped_face, 1.0 / 255.0, self.input_size, (0.0, 0.0, 0.0), swapRB=False)# blob = RGB
            
            # st = time.time()
            # inswapper expects RGB        
            if self.io_binding: 
                io_binding = self.swapper_model.io_binding()            
                io_binding.bind_cpu_input(self.input_names[0], blob)
                io_binding.bind_cpu_input(self.input_names[1], latent)
                io_binding.bind_output(self.output_names[0], "cuda")
                   
                self.swapper_model.run_with_iobinding(io_binding)
                
                ort_outs = io_binding.copy_outputs_to_cpu()
                pred = ort_outs[0]        
            else:
                pred = self.swapper_model.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
            # print(time.time() - st)
            swapped_face = pred.transpose((0,2,3,1))[0]        
            swapped_face = np.clip(255 * swapped_face, 0, 255).astype(np.float32)

        alpha = (parameters['StrengthAmount'][0]%100)*0.01
        swapped_face = swapped_face*alpha + previous_face*(1.0-alpha)

        # swapped_face = swapped_face.astype('uint8')

        
        swapped_face_upscaled = cv2.resize(swapped_face, (512,512))

        # axes = 63 - self.parameters["MaskBlur"]

        border_mask = np.zeros((128, 128), dtype=np.float32)  
         
        # border_mask = cv2.ellipse(border_mask, (63,63), (axes, int(axes/1.1)),-90, 0, 360, 1, -1)

        # sides and bottom
        top = parameters['MaskAmount'][0]
        sides = parameters['MaskAmount'][1]
        bottom = parameters['MaskAmount'][2]
        border_mask = cv2.rectangle(border_mask, (sides, top), (127-sides, 127-bottom), 255, -1)/255
        border_mask = cv2.GaussianBlur(border_mask, (parameters["MaskBlurAmount"][0]*2+1,parameters["MaskBlurAmount"][0]*2+1),0)
        img_mask = np.ones((128, 128), dtype=np.float32)  
        
        index = parameters['UpscaleMode']
        # Codeformer
        if parameters["UpscaleState"] and parameters['UpscaleModes'][index] == 'CF':           
            swapped_face_upscaled = self.codeformer(swapped_face_upscaled, parameters["UpscaleAmount"][index]) 
            
        # GFPGAN
        if parameters["UpscaleState"] and parameters['UpscaleModes'][index] == 'GFPGAN':      
            swapped_face_upscaled = self.apply_GFPGAN(swapped_face_upscaled, parameters["UpscaleAmount"][index]) 


        # Occluder
        if parameters["OccluderState"]:
            occlude_mask = self.apply_occlusion(original_face_256)
            occlude_mask = cv2.resize(occlude_mask, (128,128))  
            img_mask *= occlude_mask 

        # CLIPs CLIPs
        if parameters["CLIPState"]:
            mask = self.apply_neg_CLIPs(original_face_512, parameters["CLIPText"], parameters["CLIPAmount"][0])
            mask = cv2.resize(mask, (128,128))
            img_mask *= mask

        # Face Parsing
        if parameters["FaceParserState"]:
            mask = self.apply_bg_face_parser(swapped_face_upscaled, parameters["FaceParserAmount"][1])
            mask *= self.apply_face_parser(original_face_512, parameters["FaceParserAmount"][0])
            mask = cv2.resize(mask, (128,128))
            img_mask *= mask
            


        # Face Diffing
        if parameters["DiffState"]:
            fake_diff = self.apply_fake_diff(swapped_face, original_face, parameters["DiffAmount"][0])
            fake_diff /= 255
            img_mask *= fake_diff
        
        img_mask = cv2.GaussianBlur(img_mask, (parameters["BlurAmount"][0]*2+1,parameters["BlurAmount"][0]*2+1),0)
        img_mask *= border_mask


    
        img_mask = cv2.resize(img_mask, (512,512))
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1]) 
        swapped_face_upscaled *= img_mask
        swapped_face_upscaled = cv2.warpAffine(swapped_face_upscaled, IM512, (img.shape[1], img.shape[0]), borderValue=0.0) 

        # Option 2 - 9.8 ms
        kps_scale = 1.42

        bbox[0] = kps[2][0]-kps_ratio*56.0252*kps_scale
        bbox[1] = kps[2][1]-kps_ratio*71.7366*kps_scale
        bbox[2] = kps[2][0]+kps_ratio*71.7366*kps_scale
        bbox[3] = kps[2][1]+kps_ratio*56.0252*kps_scale

        left = floor(bbox[0])
        if left<0:
            left=0
        top = floor(bbox[1])
        if top<0: 
            top=0
        right = ceil(bbox[2])
        if right>img.shape[1]:
            right=img.shape[1]
        
        bottom = ceil(bbox[3])
        if bottom>img.shape[0]:
            bottom=img.shape[0]
        
        swapped_face_upscaled = swapped_face_upscaled[top:bottom, left:right, 0:3].astype(np.float32)  
        img_a = img[top:bottom, left:right, 0:3].astype(np.float32)
     
        img_mask = cv2.warpAffine(img_mask, IM512, (img.shape[1], img.shape[0]), borderValue=0.0)
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
        img_mask = img_mask[top:bottom, left:right, 0:1]
        

        img_mask = 1.0-img_mask 
      
        img_mask = torch.from_numpy(img_mask)
        img_a = torch.from_numpy(img_a)
        
        swapped_face_upscaled += torch.mul(img_mask,img_a).numpy()
        # swapped_face_upscaled += img_mask*img_a
        
        img[top:bottom, left:right, 0:3] = swapped_face_upscaled        
        # Option 2        
        for i in range(rot):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)        
        
        return img.astype(np.uint8)   #BGR
        
    # @profile    
    def apply_occlusion(self, img):        
        img = (img /255.0)

        img = np.float32(img[np.newaxis,:,:,:])
        img = img.transpose(0, 3, 1, 2)
        
        inputs = {"img": img}
        
        if self.io_binding: 
            io_binding = self.occluder_model.io_binding()            
            io_binding.bind_cpu_input('img', img)
            io_binding.bind_output('output', "cuda")
               
            self.occluder_model.run_with_iobinding(io_binding)
            occlude_mask = io_binding.copy_outputs_to_cpu()[0][0]
        else:
            occlude_mask = self.occluder_model.run(None, inputs)[0][0]     

        occlude_mask = (occlude_mask > 0)
        occlude_mask = occlude_mask.transpose(1, 2, 0).astype(np.float32)

        
        # occlude_mask = occlude_mask.squeeze().numpy()*1.0

        return occlude_mask         
    
      
    def apply_neg_CLIPs(self, img, CLIPText, CLIPAmount):
        clip_mask = np.ones((352, 352))
        
 
        CLIPimg = self.clip_transform(img).unsqueeze(0)

        
        if CLIPText != "":
            prompts = CLIPText.split(',')
            with lock:
                with torch.no_grad():
                    preds = self.clip_session(CLIPimg.repeat(len(prompts),1,1,1), prompts)[0]
                    # preds = self.clip_session(CLIPimg,  maskimg, True)[0]

            clip_mask = 1 - torch.sigmoid(preds[0][0])
            for i in range(len(prompts)-1):
                clip_mask *= 1-torch.sigmoid(preds[i+1][0])
            clip_mask = clip_mask.data.cpu().numpy()
            
            thresh = CLIPAmount/100.0
            clip_mask[clip_mask>thresh] = 1.0
            clip_mask[clip_mask<=thresh] = 0.0

        return clip_mask     
    # @profile
    def apply_face_parser(self, img, FaceParserAmount):

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        out = np.ones((512, 512), dtype=np.float32) 
        
        img1 = self.face_parsing_tensor(img.astype(np.uint8))
        img = torch.unsqueeze(img1, 0).numpy()      
        
        # turn mouth parser off at 0 so someone can just use the background parser
        if FaceParserAmount != 0:
            if self.io_binding:
                io_binding = self.face_parsing_model.io_binding()            
                io_binding.bind_cpu_input("input", img)
                io_binding.bind_output("out")
                   
                self.face_parsing_model.run_with_iobinding(io_binding)
                out = io_binding.copy_outputs_to_cpu()[0]
            else:
                out = self.face_parsing_model.run(None, {'input':img})[0]

            out = out.squeeze(0).argmax(0)
            
            if FaceParserAmount == -1:
                out = np.isin(out, [11]).astype('float32')
                out = -1.0*(out-1.0)
            else:
                out = np.isin(out, [11,12,13]).astype('float32')
                out = -1.0*(out-1.0)
                size = int(FaceParserAmount-1)
                kernel = np.ones((size, size))
                out = cv2.erode(out, kernel, iterations=3)
            
        return out.clip(0,1)

    def apply_bg_face_parser(self, img, FaceParserAmount):

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        out = np.ones((512, 512), dtype=np.float32)  
        
        # turn mouth parser off at 0 so someone can just use the mouth parser
        if FaceParserAmount > 0:
            img1 = self.face_parsing_tensor(img.astype(np.uint8))
            img = torch.unsqueeze(img1, 0).numpy()      

            if self.io_binding:
                io_binding = self.face_parsing_model.io_binding()            
                io_binding.bind_cpu_input("input", img)
                io_binding.bind_output("out")
                   
                self.face_parsing_model.run_with_iobinding(io_binding)
                out = io_binding.copy_outputs_to_cpu()[0]
            else:
                out = self.face_parsing_model.run(None, {'input':img})[0]

            out = out.squeeze(0).argmax(0)
            out = np.isin(out, [0, 16, 17, 18]).astype('float32')
            out = -1.0*(out-1.0)
            
            size = int(FaceParserAmount-1)
            kernel = np.ones((size, size))
            out = cv2.dilate(out, kernel, iterations=3)
        
        return out.clip(0,1)

    def apply_GFPGAN(self, swapped_face_upscaled, GFPGANAmount):
        temp = swapped_face_upscaled / 255.0

        temp[:,:,0] = (temp[:,:,0]-0.5)/0.5
        temp[:,:,1] = (temp[:,:,1]-0.5)/0.5
        temp[:,:,2] = (temp[:,:,2]-0.5)/0.5
        temp = np.float32(temp[np.newaxis,:,:,:])
        temp = temp.transpose(0, 3, 1, 2)

        ort_inputs = {"input": temp}
        if self.io_binding:
            io_binding = self.GFPGAN_model.io_binding()            
            io_binding.bind_cpu_input("input", temp)
            # io_binding.bind_output("1288", "cuda")
            io_binding.bind_output("output", "cuda")
               
            self.GFPGAN_model.run_with_iobinding(io_binding)
            ort_outs = io_binding.copy_outputs_to_cpu()
        else:
            
            ort_outs = self.GFPGAN_model.run(None, ort_inputs)
        
        output = ort_outs[0][0]

        # postprocess
        output = output.clip(-1,1)
        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
        # output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (output * 255.0).round()

        alpha = float(GFPGANAmount)/100.0
        
        swapped_face_upscaled = output*alpha + swapped_face_upscaled*(1.0-alpha)

        
        return swapped_face_upscaled
        
    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        fake_diff = swapped_face.astype(np.float32) - original_face.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2,:] = 0
        fake_diff[-2:,:] = 0
        fake_diff[:,:2] = 0
        fake_diff[:,-2:] = 0        
        
        fthresh = DiffAmount/2.0
        fake_diff[fake_diff<fthresh] = 0
        fake_diff[fake_diff>=fthresh] = 255 

        return fake_diff    
    # @profile        
    def codeformer(self, swapped_face_upscaled, GFPGANAmount):

        img =swapped_face_upscaled

        img = img.astype(np.float32)[:,:,::-1] / 255.0
        img = img.transpose((2, 0, 1))
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, axis=0).astype(np.float32)
        w = np.array([1.0], dtype=np.double)
        
        if self.io_binding: 
            io_binding = self.codeformer_model.io_binding()            
            io_binding.bind_cpu_input('x', img)
            io_binding.bind_cpu_input('w', w)
            io_binding.bind_output('y', "cuda")
               
            self.codeformer_model.run_with_iobinding(io_binding)
            output = io_binding.copy_outputs_to_cpu()[0][0]
        else:
            output = self.codeformer_model.run(None, {'x':img, 'w':w})[0][0]

        img = (output.transpose(1,2,0).clip(-1,1) + 1) * 0.5
        img = (img * 255)[:,:,::-1]
        img = img.clip(0, 255)
        alpha = float(GFPGANAmount)/100.0
        
        img = img*alpha + swapped_face_upscaled*(1.0-alpha)
        
        return img
        

