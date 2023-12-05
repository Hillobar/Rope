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

from torchvision.transforms.functional import normalize
import torch
from torchvision import transforms
from torchvision.ops import nms
import json
import math

from itertools import product as product


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
        
        # training ref
        
        self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])

        # self.faceapp_kps = self.arcface_dst * 4.0
        # self.faceapp_kps[:,0] += 32.0

        # self.FFHQM = cv2.estimateAffinePartial2D(self.faceapp_kps, self.FFHQ_kps, method = cv2.LMEDS)[0]
        # self.FFHQIM = cv2.invertAffineTransform(self.FFHQM)  
        
        self.scale_4 = cv2.getRotationMatrix2D((0,0), 0, 4.025)
        
        # for res50
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        image_size = 512
        feature_maps = [[64, 64], [32, 32], [16, 16]]
        # print(feature_maps)
        self.anchors = []
        for k, f in enumerate(feature_maps):
            min_size_array = min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_size_array:
                    s_kx = min_size / image_size
                    s_ky = min_size / image_size
                    dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                    dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        self.anchors += [cx, cy, s_kx, s_ky]        
        
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
        self.stop_marker = -1
        self.perf_test = False
        self.GFPGAN_pth = []
        self.resnet_model = []
        
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
        
    # def load_null(self):
        # self.is_video_loaded = False
        # self.is_image_loaded = False
        # self.play = False 
        # self.frame_q = []            
        # self.r_frame_q = [] 
        # self.target_facess = []
    
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

    def find_lowest_frame(self, queues):
    
        min_frame=999999999
        index=-1
        
        for idx, thread in enumerate(queues):
            frame = thread['FrameNumber']
            if frame != []:
                if frame < min_frame:
                    min_frame = frame
                    index=idx
        return index, min_frame


    def play_video(self, command):        
        if command == "play":
            self.play = True
            self.fps_average = []            
            self.process_qs = []
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            for i in range(self.num_threads):
                    new_process_q = self.process_q.copy()
                    self.process_qs.append(new_process_q)

        elif command == "stop":
            self.play = False
            self.add_action("stop_play", True)
            
            index, min_frame = self.find_lowest_frame(self.process_qs)
            
            if index != -1:
                self.current_frame = min_frame-1   


        elif command == "record":
            self.record = True
            self.play = True
            self.total_thread_time = 0.0
            self.process_qs = []
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            for i in range(self.num_threads):
                    new_process_q = self.process_q.copy()
                    self.process_qs.append(new_process_q)

           # Initialize
            self.timer = time.time()
            frame_width = int(self.capture.get(3))
            frame_height = int(self.capture.get(4))

            self.start_time = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))            
            
            self.file_name = os.path.splitext(os.path.basename(self.target_video))
            base_filename =  self.file_name[0]+"_"+str(time.time())[:10]
            self.output = os.path.join(self.saved_video_path, base_filename)
            self.temp_file = self.output+"_temp"+self.file_name[1]  

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

            index, min_frame = self.find_lowest_frame(self.process_qs)

            if index != -1:
                if self.process_qs[index]['Status'] == 'finished':
                    temp = [self.process_qs[index]['ProcessedFrame'], self.process_qs[index]['FrameNumber']]
                    self.frame_q.append(temp)

                    # Report fps, other data
                    self.fps_average.append(1.0/time_diff)
                    if len(self.fps_average) >= floor(self.fps):
                        fps = round(np.average(self.fps_average), 2)
                        msg = "%s fps, %s process time" % (fps, round(self.process_qs[index]['ThreadTime'], 4))
                        self.add_action("send_msg", msg)
                        self.fps_average = []

                    if self.process_qs[index]['FrameNumber'] >= self.video_frame_total-1 or self.process_qs[index]['FrameNumber'] == self.stop_marker:
                        self.play_video('stop')
                        
                    self.process_qs[index]['Status'] = 'clear'
                    self.process_qs[index]['Thread'] = []
                    self.process_qs[index]['FrameNumber'] = []
                    self.process_qs[index]['ThreadTime'] = []
                    self.frame_timer = time.time()
                    
        elif self.record:
           
            index, min_frame = self.find_lowest_frame(self.process_qs)           
            
            if index != -1:

                # If the swapper thread has finished generating a frame
                if self.process_qs[index]['Status'] == 'finished':
                    image = self.process_qs[index]['ProcessedFrame']  

                    pil_image = Image.fromarray(image)
                    pil_image.save(self.sp.stdin, 'JPEG')   

                    temp = [image, self.process_qs[index]['FrameNumber']]
                    self.frame_q.append(temp)

                    # Close video and process
                    if self.process_qs[index]['FrameNumber'] >= self.video_frame_total-1 or self.process_qs[index]['FrameNumber'] == self.stop_marker or self.play == False:
                        self.play_video("stop")
                        stop_time = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))
                        if stop_time == 0:
                            stop_time = float(self.video_frame_total) / float(self.fps)
                        
                        self.sp.stdin.close()
                        self.sp.wait()

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
                        self.record = False
                        msg = "Total time: %s s." % (round(timef,1))
                        print(msg)
                        self.add_action("send_msg", msg) 
                        
                    self.total_thread_time = []
                    self.process_qs[index]['Status'] = 'clear'
                    self.process_qs[index]['FrameNumber'] = []
                    self.process_qs[index]['Thread'] = []
                    self.frame_timer = time.time()
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
        orientation = int(parameters['OrientationAmount'][0]/parameters['OrientationInc'])

        for i in range(orientation):
            target_image = cv2.rotate(target_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        found_faces = self.func_w_test('faceapp', self.faceapp_model.get, target_image, max_num=10)
        
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
                        img = self.func_w_test("swap_video", self.swap_core, img, fface.kps, s_e, orientation, parameters)
                        # img = self.swap_core(img, fface.kps, s_e, orientation, parameters) 
            if not parameters['MaskViewState']:
                for i in range(orientation):
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
            target_image = img
        else:
            for i in range(orientation):
                target_image = cv2.rotate(target_image, cv2.ROTATE_90_CLOCKWISE)
        
        if self.perf_test:
            print('------------------------')        
            
        return target_image

    def findCosineDistance(self, vector1, vector2):
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

    def func_w_test(self, name, func, *args, **argsv):
        timing = time.time()
        result = func(*args, **argsv)
        if self.perf_test:
            print(name, round(time.time()-timing, 5), 's')
        return result

    # @profile    
    def swap_core(self, img, kps, s_e, rot, parameters): # img = RGB

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
        
        # Swap Face and blend according to Strength
        if parameters['StrengthState']:
            itex = ceil(parameters['StrengthAmount'][0]/100.)
        else:
            itex = 1
            
        if self.io_binding: 
            io_binding = self.swapper_model.io_binding()     

        for i in range(itex):
            previous_face = swapped_face.astype(np.uint8)
            blob = cv2.dnn.blobFromImage(swapped_face, 1.0 / 255.0, self.input_size, (0.0, 0.0, 0.0), swapRB=False)# blob = RGB

            # inswapper expects RGB        
            if self.io_binding: 
                io_binding = self.swapper_model.io_binding()            
                io_binding.bind_cpu_input(self.input_names[0], blob)
                io_binding.bind_cpu_input(self.input_names[1], latent)
                io_binding.bind_output(self.output_names[0], "cuda")
                   
                self.func_w_test("swapio", self.swapper_model.run_with_iobinding, io_binding)
               
                ort_outs = io_binding.copy_outputs_to_cpu()
                pred = ort_outs[0]        
            else:
                pred = self.func_w_test("swap", self.swapper_model.run, self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]

            swapped_face = pred.transpose((0,2,3,1))[0]     
            swapped_face = np.clip(255 * swapped_face, 0, 255)

        swapped_face = swapped_face.astype(np.float32)
    
        if parameters['StrengthState']:
            alpha = np.mod(parameters['StrengthAmount'][0], 100)
            alpha = np.multiply(alpha, 0.01)
            if alpha==0:
                alpha=1
            alpha = np.float32(alpha)

            previous_face = previous_face.astype(np.float32)
            swapped_face = np.add(np.multiply(swapped_face, alpha), np.multiply(previous_face, np.subtract(1, alpha)), dtype=np.float32)      
            # swapped_face = swapped_face*alpha + previous_face*(1-alpha)


        
        # cv2.imwrite('1.png',cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR))
        
        # convert to float32 for other models to work (codeformer)
        # swapped_face = swapped_face.astype(np.float32)
        
        
        swapped_face_upscaled = cv2.warpAffine(swapped_face, self.scale_4, (512, 512))#, flags=cv2.INTER_CUBIC)
        # swapped_face_upscaled = cv2.resize(swapped_face, (512,512), interpolation = cv2.INTER_CUBIC)
        swapped_face_upscaled = swapped_face_upscaled.clip(0, 255)
        
        
        border_mask = np.zeros((128, 128), dtype=np.float32)  
        # border_mask = cv2.ellipse(border_mask, (63,63), (axes, int(axes/1.1)),-90, 0, 360, 1, -1)

        # sides and bottom
        top = parameters['BorderAmount'][0]
        sides = parameters['BorderAmount'][1]
        bottom = parameters['BorderAmount'][2]
        blur = parameters['BorderAmount'][3]
        border_mask = cv2.rectangle(border_mask, (sides, top), (127-sides, 127-bottom), 255, -1)/255
        border_mask = cv2.GaussianBlur(border_mask, (blur*2+1,blur*2+1),0)
        img_mask = np.ones((128, 128), dtype=np.float32)  


        # Codeformer
        if parameters["UpscaleState"] and parameters['UpscaleMode']==1:   
            swapped_face_upscaled = self.func_w_test('codeformer', self.apply_codeformer, swapped_face_upscaled, parameters["UpscaleAmount"][1])

        # GFPGAN
        if parameters["UpscaleState"] and parameters['UpscaleMode']==0: 
            swapped_face_upscaled = self.func_w_test('GFPGAN_onnx', self.apply_GFPGAN, swapped_face_upscaled, parameters["UpscaleAmount"][0])
  
        # Occluder
        if parameters["OccluderState"]:
            mask = self.func_w_test('occluder', self.apply_occlusion , original_face_256)
            mask = cv2.resize(mask, (128,128))  
            img_mask *= mask 

        # CLIPs CLIPs
        if parameters["CLIPState"]:
            with lock:
                mask = self.func_w_test('CLIP', self.apply_neg_CLIPs, original_face_512, parameters["CLIPText"], parameters["CLIPAmount"][0])
            mask = cv2.resize(mask, (128,128))
            img_mask *= mask

        # Face Parsing
        if parameters["FaceParserState"]:
            mask = self.func_w_test('bg parser', self.apply_bg_face_parser, swapped_face_upscaled, parameters["FaceParserAmount"][1])
            mask *= self.func_w_test('mouth parser', self.apply_face_parser, original_face_512, parameters["FaceParserAmount"][0])
            mask = cv2.resize(mask, (128,128))
            img_mask *= mask
            


        # Face Diffing
        if parameters["DiffState"]:
            mask = self.apply_fake_diff(swapped_face, original_face, parameters["DiffAmount"][0])
            mask /= 255
            img_mask *= mask
        
        img_mask = cv2.GaussianBlur(img_mask, (parameters["BlurAmount"][0]*2+1,parameters["BlurAmount"][0]*2+1),0)
        img_mask *= border_mask
    
        img_mask = cv2.warpAffine(img_mask, self.scale_4, (512, 512))#, flags=cv2.INTER_CUBIC)
        # img_mask = cv2.resize(img_mask, (512,512))
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1]) 
        img_mask = img_mask.clip(0, 1)
        
        swapped_face_upscaled *= img_mask

        if not parameters['MaskViewState']:
        
            swapped_face_upscaled = cv2.warpAffine(swapped_face_upscaled, IM512, (img.shape[1], img.shape[0]), borderValue=0.0) 

            # Option 2 - 9.8 ms
            kps_scale = 1.42
            bbox = [0]*4
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
            img[top:bottom, left:right, 0:3] = swapped_face_upscaled        

        else:
            img_mask = cv2.merge((img_mask, img_mask, img_mask))
            swapped_face_upscaled += (1.0-img_mask)*original_face_512
            img = np.hstack([swapped_face_upscaled, img_mask*255])


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
        # turn mouth parser off at 0 so someone can just use the background parser
        if FaceParserAmount != 0:        
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
            
            if FaceParserAmount <0:
                out = np.isin(out, [11]).astype('float32')
                out = -1.0*(out-1.0)
                size = int(-FaceParserAmount)
                kernel = np.ones((size, size))
                out = cv2.erode(out, kernel, iterations=2)
            elif FaceParserAmount >0:
                out = np.isin(out, [11,12,13]).astype('float32')
                out = -1.0*(out-1.0)
                size = int(FaceParserAmount)
                kernel = np.ones((size, size))
                out = cv2.erode(out, kernel, iterations=2)
            
        return out.clip(0,1)

    def apply_bg_face_parser(self, img, FaceParserAmount):

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        out = np.ones((512, 512), dtype=np.float32)  
        
        # turn mouth parser off at 0 so someone can just use the mouth parser
        if FaceParserAmount != 0:
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

            if FaceParserAmount >0:                   
                size = int(FaceParserAmount)
                kernel = np.ones((size, size))
                out = cv2.dilate(out, kernel, iterations=2)
            elif FaceParserAmount <0:
                size = int(-FaceParserAmount)
                kernel = np.ones((size, size))
                out = cv2.erode(out, kernel, iterations=2)
        return out.clip(0,1)

    def apply_GFPGAN(self, swapped_face_upscaled, GFPGANAmount):     
        try:
            landmark = self.ret50_landmarks(swapped_face_upscaled) 
        except:
            return swapped_face_upscaled     
        
        FFHQM = cv2.estimateAffinePartial2D(landmark, self.FFHQ_kps, method = cv2.LMEDS)[0]
        FFHQIM = cv2.invertAffineTransform(FFHQM)   
        
        temp = cv2.warpAffine(swapped_face_upscaled, FFHQM, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
        temp = temp / 255.0
        temp = torch.from_numpy(temp.transpose(2, 0, 1))
        normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        temp = np.float32(temp[np.newaxis,:,:,:])

        ort_inputs = {"input": temp}

        if self.io_binding:
            io_binding = self.GFPGAN_model.io_binding()            
            io_binding.bind_cpu_input("input", temp)
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
        output = (output * 255.0).round()
        output = cv2.warpAffine(output, FFHQIM, (512, 512))
        
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
    
     
    def apply_codeformer(self, swapped_face_upscaled, GFPGANAmount):
        try:
            landmark = self.ret50_landmarks(swapped_face_upscaled) 
        except:
            return swapped_face_upscaled
        
        FFHQM = cv2.estimateAffinePartial2D(landmark, self.FFHQ_kps, method = cv2.LMEDS)[0]
        FFHQIM = cv2.invertAffineTransform(FFHQM)         
    
    
        img = cv2.warpAffine(swapped_face_upscaled, FFHQM, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))         
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.warpAffine(img, FFHQIM, (512, 512))

        alpha = float(GFPGANAmount)/100.0
        img = img*alpha + swapped_face_upscaled*(1.0-alpha)
        
        return img
        
    

    # @profile    
    def ret50_landmarks(self, image):    
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = image - [104., 117., 123.]
        image = image.transpose(2, 0, 1)
        image = np.float32(image[np.newaxis,:,:,:])

        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device='cuda')
        
        ort_inputs = {"input": image}        
        _, conf, landmarks = self.resnet_model.run(None, ort_inputs)

        conf = torch.from_numpy(conf)
        scores = conf.squeeze(0).numpy()[:, 1]
        
        landmarks = torch.from_numpy(landmarks)
        landmarks = landmarks.to('cuda')        

        priors = torch.Tensor(self.anchors).view(-1, 4)
        priors = priors.to('cuda')

        pre = landmarks.squeeze(0) 
        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)
        landmarks = landmarks * scale1
        landmarks = landmarks.cpu().numpy()  

        # ignore low scores
        inds = np.where(scores > 0.97)[0]
        landmarks, scores = landmarks[inds], scores[inds]    

        # sort
        order = scores.argsort()[::-1]
        landmarks = landmarks[order][0]

        return np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])
        
    # def ret50_landmarks(self, image):    
        # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # image = image - [104., 117., 123.]
        # image = image.transpose(2, 0, 1)
        # image = np.float32(image[np.newaxis,:,:,:])

        # height, width = (512, 512)
        # scale = torch.tensor([width, height, width, height], dtype=torch.float32, device='cuda')
        # tmp = [width, height, width, height, width, height, width, height, width, height]
        # scale1 = torch.tensor(tmp, dtype=torch.float32, device='cuda')
        
        # ort_inputs = {"input": image}    
    
        # loc, conf, landmarks = self.resnet_model.run(None, ort_inputs)
    
        # loc = torch.from_numpy(loc)
        # loc = loc.to('cuda')
        
        # conf = torch.from_numpy(conf)
        # conf = conf.to('cuda')
        
        # landmarks = torch.from_numpy(landmarks)
        # landmarks = landmarks.to('cuda')        
        
        # # priors = self.priors(image.shape[2:])
        
        # min_sizes = [[16, 32], [64, 128], [256, 512]]
        # steps = [8, 16, 32]
        # image_size = image.shape[2:]
        # feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]

        # anchors = []
        # for k, f in enumerate(feature_maps):
            # min_size_array = min_sizes[k]
            # for i, j in product(range(f[0]), range(f[1])):
                # for min_size in min_size_array:
                    # s_kx = min_size / image_size[1]
                    # s_ky = min_size / image_size[0]
                    # dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                    # dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                    # for cy, cx in product(dense_cy, dense_cx):
                        # anchors += [cx, cy, s_kx, s_ky]

        # # back to torch land
        # priors = torch.Tensor(anchors).view(-1, 4)
        # priors = priors.to('cuda')

        # # boxes = self.decode(loc.data.squeeze(0), priors.data)
        # loc = loc.squeeze(0)
        # boxes = torch.cat((priors[:, :2] + loc[:, :2] * 0.1 * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), 1)
        # boxes[:, :2] -= boxes[:, 2:] / 2
        # boxes[:, 2:] += boxes[:, :2]
        # boxes = boxes * scale
        # boxes = boxes.cpu().numpy()
        
        # scores = conf.squeeze(0).cpu().numpy()[:, 1]
        
        # pre = landmarks.squeeze(0) 
        # tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        # landmarks = torch.cat(tmp, dim=1)
        # landmarks = landmarks * scale1
        # landmarks = landmarks.cpu().numpy()  

        # # ignore low scores
        # inds = np.where(scores > 0.97)[0]
        # boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]    

        # # sort
        # order = scores.argsort()[::-1]
        # boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]  

        # # do NMS
        # bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = list(nms(boxes=torch.Tensor(bounding_boxes[:, :4]), scores=torch.Tensor(bounding_boxes[:, 4]), iou_threshold=0.4))
        
        # bounding_boxes, landmarks = bounding_boxes[keep, :], landmarks[keep]        
        
        # bbox = np.concatenate((bounding_boxes, landmarks), axis=1)
        # bbox = bbox[0]
        # #freaem 1046
        # return np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])  