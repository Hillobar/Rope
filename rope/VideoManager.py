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

import onnxruntime

import torchvision

from torchvision.transforms.functional import normalize #update to v2
# from torchvision.transforms import v2
import torch
from torchvision import transforms
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchvision.ops import nms
import json
import math

from itertools import product as product
# cv2.cuda.createGpuMatFromCudaMemory(h, w, cv2.CV_32F, x.data_ptr())
device = 'cuda'
onnxruntime.set_default_logger_severity(4)

# from itertools import combinations

lock=threading.Lock()

class VideoManager():  
    def __init__( self ):
        # Model related
        self.swapper_model = []             # insightface swapper model
        # self.faceapp_model = []             # insight faceapp model
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
        self.GPEN_256_model = []
        self.GPEN_512_model = []

        self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])
        
      
        # for res50
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        image_size = 512
        feature_maps = [[64, 64], [32, 32], [16, 16]]

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
        # self.source_embedding = []          # array with indexed source embeddings
        self.swap = False                   # flag for the swap enabled toggle
        self.target_facess = []   # array that maps the found faces to source faces    

        self.parameters = []

        self.num_threads = 0
        self.target_video = []

        self.fps = 1.0
        self.temp_file = []

        # self.i_image = []
        self.io_binding = True
        # self.video_read_success = False
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

        self.resnet_model = []
        self.detection_model = []
        self.recognition_model = []
        self.syncvec = torch.empty((1,1), dtype=torch.float32, device=device)

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

            # args =  ["ffmpeg", 
                    # '-hide_banner',
                    # '-loglevel',    'error',
                    # "-an",       
                    # "-r",           str(self.fps),
                    # "-i",           "pipe:",
                    # # '-g',           '25',
                    # "-vf",          "format=yuvj420p",
                    # "-c:v",         "libx264",
                    # "-crf",         str(self.vid_qual),
                    # "-r",           str(self.fps),
                    # "-s",           str(frame_width)+"x"+str(frame_height),
                    # self.temp_file]  
            
            # self.sp = subprocess.Popen(args, stdin=subprocess.PIPE)
            size = (frame_width, frame_height)
            self.sp = cv2.VideoWriter(self.temp_file,  cv2.VideoWriter_fourcc(*'mp4v') , self.fps, size) 
      
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

                    # pil_image = Image.fromarray(image)
                    # pil_image.save(self.sp.stdin, 'JPEG')   
                    self.sp.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    temp = [image, self.process_qs[index]['FrameNumber']]
                    self.frame_q.append(temp)

                    # Close video and process
                    if self.process_qs[index]['FrameNumber'] >= self.video_frame_total-1 or self.process_qs[index]['FrameNumber'] == self.stop_marker or self.play == False:
                        self.play_video("stop")
                        stop_time = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))
                        if stop_time == 0:
                            stop_time = float(self.video_frame_total) / float(self.fps)
                        
                        # self.sp.stdin.close()
                        # self.sp.wait()
                        self.sp.release()

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

    # @profile
    def swap_video(self, target_image, frame_number, change_parameters):   
        # Grab a local copy of the parameters to prevent threading issues
        parameters = self.parameters.copy()
        
        # Find out if the frame is in a marker zone and copy the parameters if true
        if self.markers and not change_parameters:
            temp=[]
            for i in range(len(self.markers)):
                temp.append(self.markers[i]['frame'])
            idx = bisect.bisect(temp, frame_number)
            
            parameters = self.markers[idx-1]['parameters'].copy()
        
        # Load frame into VRAM
        img = torch.from_numpy(target_image).to('cuda') #HxWxc
        img = img.permute(2,0,1)#cxHxW        
        
        #Scale up frame if it is smaller than 512
        img_x = img.size()[2]
        img_y = img.size()[1]
        
        if img_x<512 and img_y<512:
            # if x is smaller, set x to 512
            if img_x <= img_y:
                tscale = v2.Resize((int(512*img_y/img_x), 512), antialias=True)
            else:
                tscale = v2.Resize((512, int(512*img_x/img_y)), antialias=True)

            img = tscale(img)
            
        elif img_x<512:
            tscale = v2.Resize((int(512*img_y/img_x), 512), antialias=True)
            img = tscale(img)
        
        elif img_y<512:
            tscale = v2.Resize((512, int(512*img_x/img_y)), antialias=True)
            img = tscale(img)    

        # Rotate the frame
        if parameters['OrientationState']:
            img = transforms.functional.rotate(img, angle=parameters['OrientationAmount'][0], expand=True)

        # Find all faces in frame and return a list of 5-pt kpss
        kpss = self.func_w_test("detect", self.detect, img, input_size = (640, 640), max_num=10, metric='default')
        
        # Get embeddings for all faces found in the fram
        ret = []
        for i in range(kpss.shape[0]):
            if kpss is not None:
                face_kps = kpss[i]

            face_emb = self.func_w_test('recognize',  self.recognize, img, face_kps)
            ret.append([face_kps, face_emb])
        
        if ret:
            # Loop through target faces to see if they match our target embeddings
            for fface in ret:
                for tface in self.target_facess:
                    # sim between face in video and already found face
                    sim = self.findCosineDistance(fface[1], tface["Embedding"])
                    # if the face[i] in the frame matches afound face[j] AND the found face is active (not []) 
                    threshhold = parameters["ThresholdAmount"][0]/100.0
                    if parameters["ThresholdState"]:
                        threshhold = 2.0
    
                    if sim<float(threshhold) and tface["SourceFaceAssignments"]:
                        s_e =  tface["AssignedEmbedding"]
                        img = self.func_w_test("swap_video", self.swap_core, img, fface[0], s_e, parameters, frame_number)
                        # img = img.permute(2,0,1)
                    
            img = img.permute(1,2,0)
            if not parameters['MaskViewState'] and parameters['OrientationState']:
                img = img.permute(2,0,1)
                img = transforms.functional.rotate(img, angle=-parameters['OrientationAmount'][0], expand=True)
                img = img.permute(1,2,0)

        else:
            img = img.permute(1,2,0)
            if parameters['OrientationState']:
                img = img.permute(2,0,1)
                img = transforms.functional.rotate(img, angle=-parameters['OrientationAmount'][0], expand=True)
                img = img.permute(1,2,0)
        
        if self.perf_test:
            print('------------------------')  
        
        # Unscale small videos
        if img_x <512 or img_y < 512:
            tscale = v2.Resize((img_y, img_x), antialias=True)
            img = img.permute(2,0,1)
            img = tscale(img)
            img = img.permute(1,2,0)

        img = img.cpu().numpy()    
        return img.astype(np.uint8)

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
    def swap_core(self, img, kps, s_e, parameters, frame): # img = RGB
        # 512 transforms
        dst = self.arcface_dst * 4.0
        dst[:,0] += 32.0
        
        # Change the ref points
        if parameters['RefDelState']:
            dst[:,0] += parameters['RefDelAmount'][1]
            dst[:,1] += parameters['RefDelAmount'][0]
            dst[:,0] -= 255
            dst[:,0] *= (1+parameters['RefDelAmount'][2]/100)
            dst[:,0] += 255
            dst[:,1] -= 255
            dst[:,1] *= (1+parameters['RefDelAmount'][2]/100)
            dst[:,1] += 255

        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst) 

        # Scaling Transforms
        t512 = v2.Resize((512, 512), antialias=True)
        t256 = v2.Resize((256, 256), antialias=True)
        t128 = v2.Resize((128, 128), antialias=True)

        # Grab 512 face from image and create 256 and 128 copys
        original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=v2.InterpolationMode.BILINEAR ) 

        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)# 3, 512, 512
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)  

        # Optional Scaling # change the thransform matrix
        if parameters['TransformState']:
            original_face_128 = v2.functional.affine(original_face_128, 0, (0,0) , 1+parameters['TransformAmount'][0]/100, 0, center = (63,63), interpolation=v2.InterpolationMode.BILINEAR) 

        #Normalize source embedding
        n_e = s_e / l2norm(s_e)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        latent = torch.from_numpy(latent).float().to('cuda')

        # Prepare for swapper formats
        prev_swap = torch.reshape(original_face_128, (1, 3, 128, 128))
        prev_swap = torch.div(prev_swap, 255)
        swap = prev_swap.contiguous()

        # Swap Face and blend according to Strength
        itex = 1
        if parameters['StrengthState']:
            itex = ceil(parameters['StrengthAmount'][0]/100.)

        # # Bindings
        io_binding = self.swapper_model.io_binding() 

        # Additional swaps based on strength
        for i in range(itex):
            prev_swap = swap.detach().clone()
        
            # Rebind previous output to input
            io_binding.bind_input(name=self.input_names[0], device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=prev_swap.data_ptr())
            io_binding.bind_input(name=self.input_names[1], device_type='cuda', device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=latent.data_ptr())
            io_binding.bind_output(name=self.output_names[0], device_type='cuda', device_id=0, element_type=np.float32, shape=tuple(swap.shape), buffer_ptr=swap.data_ptr())
            
            # Sync and run model
            syncvec = self.syncvec.cpu()          
            self.swapper_model.run_with_iobinding(io_binding)
            
        if parameters['StrengthState']:
            alpha = np.mod(parameters['StrengthAmount'][0], 100)*0.01
            if alpha==0:
                alpha=1
            # Blend the images
            swap = torch.mul(swap, alpha)
            prev_swap = torch.mul(prev_swap, 1-alpha)
            swap = torch.add(swap, prev_swap)

        # Format to 3x128x128 [0..255] uint8
        swap = torch.squeeze(swap)
        swap = torch.mul(swap, 255) # should I carry [0..1] through the pipe insteadf?
        swap = torch.clamp(swap, 0, 255)
        swap = swap.type(torch.uint8)
        swap_128 = swap
        swap = t512(swap)

        # Create border mask
        border_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
        border_mask = torch.unsqueeze(border_mask,0)

        top = parameters['BorderAmount'][0]
        left = parameters['BorderAmount'][1]
        right = 128-parameters['BorderAmount'][1]
        bottom = 128-parameters['BorderAmount'][2]

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        gauss = transforms.GaussianBlur(parameters['BorderAmount'][3]*2+1, (parameters['BorderAmount'][3]+1)*0.2)
        border_mask = gauss(border_mask)        

        # Create image mask
        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
        swap_mask = torch.unsqueeze(swap_mask,0)       

        # Codeformer
        if parameters["UpscaleState"] and parameters['UpscaleMode']==1:   
            swap = self.func_w_test('Codeformer', self.apply_codeformer, swap, parameters)

        # GFPGAN
        if parameters["UpscaleState"] and parameters['UpscaleMode']==0: 
            swap = self.func_w_test('GFPGAN', self.apply_GFPGAN, swap, parameters)
        
        # GPEN_256   
        if parameters["UpscaleState"] and parameters['UpscaleMode']==2: 
            GPEN_resize = t256(swap)
            swap = self.func_w_test('GPEN_256', self.apply_GPEN_256, swap, parameters)
            swap = t512(swap)

        # GPEN_512
        if parameters["UpscaleState"] and parameters['UpscaleMode']==3: 
            swap = self.func_w_test('GPEN_512', self.apply_GPEN_512, swap, parameters)
            
        # Occluder
        if parameters["OccluderState"]:
            mask = self.func_w_test('occluder', self.apply_occlusion , original_face_256, parameters["OccluderAmount"][0])
            mask = t128(mask)  
            swap_mask = torch.mul(swap_mask, mask)

        # CLIPs CLIPs
        if parameters["CLIPState"]:
            inface = original_face_512.permute(1,2,0)
            inface = inface.cpu().numpy()
            with lock:
                mask = self.func_w_test('CLIP', self.apply_neg_CLIPs, inface, parameters["CLIPText"], parameters["CLIPAmount"][0])
            mask = cv2.resize(mask, (128,128))
            mask = torch.from_numpy(mask).to('cuda')
            swap_mask *= mask

        # Face Parsing
        if parameters["FaceParserState"]:
            mask = self.func_w_test('bg parser', self.apply_bg_face_parser, swap, parameters["FaceParserAmount"][1])
            mask2 = self.func_w_test('mouth parser', self.apply_face_parser, original_face_512, parameters["FaceParserAmount"][0])
            mask = torch.mul(mask, mask2)
            mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)

        # Face Diffing
        if parameters["DiffState"]:
            mask = self.apply_fake_diff(swap_128, original_face_128, parameters["DiffAmount"][0])
            # mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)

        # Add blur to swap_mask results
        gauss = transforms.GaussianBlur(parameters['BlurAmount'][0]*2+1, (parameters['BlurAmount'][0]+1)*0.2)
        swap_mask = gauss(swap_mask)  
        
        # Apply color corerctions
        if parameters['ColorState']:
            swap = swap.permute(1, 2, 0).type(torch.float32)
            del_color = torch.tensor([parameters['ColorAmount'][0], parameters['ColorAmount'][1], parameters['ColorAmount'][2]], device=device)
            swap += del_color
            swap = torch.clamp(swap, min=0., max=255.)
            swap = swap.permute(2, 0, 1).type(torch.uint8)

        # Combine border and swap mask, scale, and apply to swap
        swap_mask = torch.mul(swap_mask, border_mask)
        swap_mask = t512(swap_mask)
        swap = torch.mul(swap, swap_mask)

        if not parameters['MaskViewState']:
            # Cslculate the area to be mergerd back to the original frame
            IM512 = tform.inverse.params[0:2, :]
            corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])

            x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
            y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])
            
            left = floor(np.min(x))
            if left<0:
                left=0
            top = floor(np.min(y))
            if top<0: 
                top=0
            right = ceil(np.max(x))
            if right>img.shape[2]:
                right=img.shape[2]            
            bottom = ceil(np.max(y))
            if bottom>img.shape[1]:
                bottom=img.shape[1]   

            # Untransform the swap
            swap = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
            swap = v2.functional.affine(swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0,interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )  
            swap = swap[0:3, top:bottom, left:right]
            swap = swap.permute(1, 2, 0)
            
            # Untransform the swap mask
            swap_mask = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
            swap_mask = v2.functional.affine(swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) ) 
            swap_mask = swap_mask[0:1, top:bottom, left:right]                        
            swap_mask = swap_mask.permute(1, 2, 0)
            swap_mask = torch.sub(1, swap_mask) 

            # Apply the mask to the original image areas
            img_crop = img[0:3, top:bottom, left:right]
            img_crop = img_crop.permute(1,2,0)            
            img_crop = torch.mul(swap_mask,img_crop)
            
            #Add the cropped areas and place them back into the original image
            swap = torch.add(swap, img_crop)
            swap = swap.type(torch.uint8)
            swap = swap.permute(2,0,1)
            img[0:3, top:bottom, left:right] = swap  

        else:
            # Invert swap mask
            swap_mask = torch.sub(1, swap_mask)
            
            # Combine preswapped face with swap
            original_face_512 = torch.mul(swap_mask, original_face_512)
            original_face_512 = torch.add(swap, original_face_512)            
            original_face_512 = original_face_512.type(torch.uint8)
            original_face_512 = original_face_512.permute(1, 2, 0)

            # Uninvert and create image from swap mask
            swap_mask = torch.sub(1, swap_mask) 
            swap_mask = torch.cat((swap_mask,swap_mask,swap_mask),0)
            swap_mask = swap_mask.permute(1, 2, 0)

            # Place them side by side
            img = torch.hstack([original_face_512, swap_mask*255])
            img = img.permute(2,0,1)

        return img
        
    # @profile    
    def apply_occlusion(self, img, amount):        
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0)

        outpred = torch.ones((256,256), dtype=torch.float32, device=device).contiguous()
        
        io_binding = self.occluder_model.io_binding()            
        io_binding.bind_input(name='img', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=img.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=outpred.data_ptr())   

        # Sync and run model
        syncvec = self.syncvec.cpu()       
        self.occluder_model.run_with_iobinding(io_binding)    
        
        outpred = torch.squeeze(outpred)

        outpred = (outpred > 0)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)
        
        if amount >0:                   
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

            for i in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
            
            outpred = torch.squeeze(outpred)
            
        if amount <0:      
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

            for i in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
            
            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            
        outpred = torch.reshape(outpred, (1, 256, 256)) 
        return outpred         
    
      
    def apply_neg_CLIPs(self, img, CLIPText, CLIPAmount):
        clip_mask = np.ones((352, 352))
        
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                                        transforms.Resize((352, 352))])
        CLIPimg = transform(img).unsqueeze(0)
        
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
       
       # out = np.ones((512, 512), dtype=np.float32) 
        outpred = torch.ones((512,512), dtype=torch.float32, device='cuda').contiguous()
        
        # turn mouth parser off at 0 so someone can just use the background parser
        if FaceParserAmount != 0:        
            img = torch.div(img, 255)
            img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            img = torch.reshape(img, (1, 3, 512, 512))      
            outpred = torch.empty((1,19,512,512), dtype=torch.float32, device='cuda').contiguous()
            
            io_binding = self.face_parsing_model.io_binding()            
            io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=img.data_ptr())
            io_binding.bind_output(name='out', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=outpred.data_ptr())   

            torch.cuda.synchronize('cuda')       
            self.face_parsing_model.run_with_iobinding(io_binding)

            outpred = torch.squeeze(outpred)
            outpred = torch.argmax(outpred, 0)
            
            if FaceParserAmount <0:
                test = torch.tensor([11], device='cuda')
                iters = int(-FaceParserAmount)
                
            elif FaceParserAmount >0:
                test = torch.tensor([11,12,13], device='cuda')
                iters = int(FaceParserAmount)
            
            outpred = torch.isin(outpred, test)            
            outpred = torch.clamp(~outpred, 0, 1).type(torch.float32)
            outpred = torch.reshape(outpred, (1,1,512,512))            
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device='cuda')

            for i in range(iters):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
                
            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
        outpred = torch.reshape(outpred, (1, 512, 512))   
    
        return outpred

    def apply_bg_face_parser(self, img, FaceParserAmount):

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        # out = np.ones((512, 512), dtype=np.float32)  
        
        outpred = torch.ones((512,512), dtype=torch.float32, device='cuda').contiguous()

        # turn mouth parser off at 0 so someone can just use the mouth parser
        if FaceParserAmount != 0:
            img = torch.div(img, 255)
            img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            img = torch.reshape(img, (1, 3, 512, 512))      
            outpred = torch.empty((1,19,512,512), dtype=torch.float32, device=device).contiguous()
            
            io_binding = self.face_parsing_model.io_binding()            
            io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=img.data_ptr())
            io_binding.bind_output(name='out', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=outpred.data_ptr())   

            torch.cuda.synchronize('cuda')       
            self.face_parsing_model.run_with_iobinding(io_binding)

            outpred = torch.squeeze(outpred)
            outpred = torch.argmax(outpred, 0)

            test = torch.tensor([ 0, 14, 15, 16, 17, 18], device=device)
            outpred = torch.isin(outpred, test)  
            outpred = torch.clamp(~outpred, 0, 1).type(torch.float32)            
            outpred = torch.reshape(outpred, (1,1,512,512))
            
            if FaceParserAmount >0:                   
                kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

                for i in range(int(FaceParserAmount)):
                    outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                    outpred = torch.clamp(outpred, 0, 1)
                
                outpred = torch.squeeze(outpred)
                
            if FaceParserAmount <0:      
                outpred = torch.neg(outpred)
                outpred = torch.add(outpred, 1)

                kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

                for i in range(int(-FaceParserAmount)):
                    outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                    outpred = torch.clamp(outpred, 0, 1)
                
                outpred = torch.squeeze(outpred)
                outpred = torch.neg(outpred)
                outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 512, 512))
        
        return outpred
    
    # @profile
    def apply_GPEN_256(self, swapped_face_upscaled, parameters):     
        # Set up Transformation
        dst = self.arcface_dst * 4.0
        dst[:,0] += 32.0        
        tform = trans.SimilarityTransform()   

        t512 = v2.Resize((512, 512), antialias=True)
        t256 = v2.Resize((256, 256), antialias=True)         
        
        if self.is_image_loaded:
            try:
                dst = self.ret50_landmarks(swapped_face_upscaled) 
            except:
                return swapped_face_upscaled       

        tform.estimate(dst, self.FFHQ_kps)

        # Transform, scale, and normalize
        temp = v2.functional.affine(swapped_face_upscaled, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
        temp = v2.functional.crop(temp, 0,0, 512, 512)        
        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
        temp = t256(temp)
        temp = torch.unsqueeze(temp, 0)

        # Bindings
        outpred = torch.empty((1,3,256,256), dtype=torch.float32, device=device).contiguous()
        io_binding = self.GPEN_256_model.io_binding() 
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=temp.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=outpred.data_ptr())
        
        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.GPEN_256_model.run_with_iobinding(io_binding)
        
        # Format back to cxHxW @ 255
        outpred = torch.squeeze(outpred)      
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)
        outpred = t512(outpred)
        
        # Invert Transform
        outpred = v2.functional.affine(outpred, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]) , tform.
        inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )

        # Blend
        alpha = float(parameters["UpscaleAmount"][2])/100.0  
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred                   
             
                
    def apply_GPEN_512(self, swapped_face_upscaled, parameters):     
        # Set up Transformation
        dst = self.arcface_dst * 4.0
        dst[:,0] += 32.0        
        tform = trans.SimilarityTransform()        
        
        if self.is_image_loaded:
            try:
                dst = self.ret50_landmarks(swapped_face_upscaled) 
            except:
                return swapped_face_upscaled       

        tform.estimate(dst, self.FFHQ_kps)

        # Transform, scale, and normalize
        temp = v2.functional.affine(swapped_face_upscaled, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
        temp = v2.functional.crop(temp, 0,0, 512, 512)        
        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
        temp = torch.unsqueeze(temp, 0)

        # Bindings
        outpred = torch.empty((1,3,512,512), dtype=torch.float32, device=device).contiguous()
        io_binding = self.GPEN_512_model.io_binding() 
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=temp.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=outpred.data_ptr())
        
        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.GPEN_512_model.run_with_iobinding(io_binding)
        
        # Format back to cxHxW @ 255
        outpred = torch.squeeze(outpred)      
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)
        
        # Invert Transform
        outpred = v2.functional.affine(outpred, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]) , tform.
        inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )

        # Blend
        alpha = float(parameters["UpscaleAmount"][3])/100.0  
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred                
                
    def apply_GFPGAN(self, swapped_face_upscaled, parameters):     
        # Set up Transformation
        dst = self.arcface_dst * 4.0
        dst[:,0] += 32.0        
        tform = trans.SimilarityTransform()        
        
        if self.is_image_loaded:
            try:
                dst = self.ret50_landmarks(swapped_face_upscaled) 
            except:
                return swapped_face_upscaled       

        tform.estimate(dst, self.FFHQ_kps)

        # Transform, scale, and normalize
        temp = v2.functional.affine(swapped_face_upscaled, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
        temp = v2.functional.crop(temp, 0,0, 512, 512)        
        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
        temp = torch.unsqueeze(temp, 0)

        # Bindings
        outpred = torch.empty((1,3,512,512), dtype=torch.float32, device=device).contiguous()
        io_binding = self.GFPGAN_model.io_binding() 
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=temp.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=outpred.data_ptr())
        
        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.GFPGAN_model.run_with_iobinding(io_binding)
        
        # Format back to cxHxW @ 255
        outpred = torch.squeeze(outpred)      
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)
        
        # Invert Transform
        outpred = v2.functional.affine(outpred, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]) , tform.
        inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )

        # Blend
        alpha = float(parameters["UpscaleAmount"][0])/100.0  
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred
        
    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        swapped_face = swapped_face.permute(1,2,0)
        original_face = original_face.permute(1,2,0)

        diff = swapped_face-original_face
        diff = torch.abs(diff)
        
        # Find the diffrence between the swap and original, per channel
        fthresh = DiffAmount*2.55
        
        # Bimodal
        diff[diff<fthresh] = 0
        diff[diff>=fthresh] = 1 
        
        # If any of the channels exceeded the threshhold, them add them to the mask
        diff = torch.sum(diff, dim=2)
        diff = torch.unsqueeze(diff, 2)
        diff[diff>0] = 1
        
        diff = diff.permute(2,0,1)

        return diff    
    
    def apply_codeformer(self, swapped_face_upscaled, parameters):
        # Set up Transformation
        dst = self.arcface_dst * 4.0
        dst[:,0] += 32.0        
        tform = trans.SimilarityTransform()        
        
        # # Select detection approach
        if self.is_image_loaded:
            try:
                dst = self.ret50_landmarks(swapped_face_upscaled) 
            except:
                return swapped_face_upscaled     

        tform.estimate(dst, self.FFHQ_kps)        
 
        # Transform, scale, and normalize
        temp = v2.functional.affine(swapped_face_upscaled, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
        temp = v2.functional.crop(temp, 0,0, 512, 512)   
        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
        temp = torch.reshape(temp, (1, 3, 512, 512))#############change to unsqueeze

        # Bindings
        outpred = torch.empty((1,3,512,512), dtype=torch.float32, device=device).contiguous()
        w = np.array([1.0], dtype=np.double)
        io_binding = self.codeformer_model.io_binding() 
        io_binding.bind_input(name='x', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=temp.data_ptr())
        io_binding.bind_cpu_input('w', w)
        io_binding.bind_output(name='y', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=outpred.data_ptr())
        
        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.codeformer_model.run_with_iobinding(io_binding)           

        # Format back to cxHxW @ 255
        outpred = torch.squeeze(outpred)      
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)

        # Invert Transform
        outpred = v2.functional.affine(outpred, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]) , tform.
        inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )

        # Blend
        alpha = float(parameters["UpscaleAmount"][1])/100.0  
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred

    # @profile    
    def ret50_landmarks(self, image):    
        # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = image.permute(1,2,0)
        
        # image = image - [104, 117, 123]
        mean = torch.tensor([104, 117, 123], dtype=torch.float32, device='cuda')
        image = torch.sub(image, mean)
        
        # image = image.transpose(2, 0, 1)
        # image = np.float32(image[np.newaxis,:,:,:])
        image = image.permute(2,0,1)
        image = torch.reshape(image, (1, 3, 512, 512))


        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device='cuda')
        
        # ort_inputs = {"input": image}        
        conf = torch.empty((1,10752,2), dtype=torch.float32, device=device).contiguous()
        landmarks = torch.empty((1,10752,10), dtype=torch.float32, device=device).contiguous()

        io_binding = self.resnet_model.io_binding() 
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='conf', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,2), buffer_ptr=conf.data_ptr())
        io_binding.bind_output(name='landmarks', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,10), buffer_ptr=landmarks.data_ptr())
        
        # _, conf, landmarks = self.resnet_model.run(None, ort_inputs)        
        torch.cuda.synchronize('cuda')
        self.resnet_model.run_with_iobinding(io_binding)        
        

        # conf = torch.from_numpy(conf)
        # scores = conf.squeeze(0).numpy()[:, 1]
        scores = torch.squeeze(conf)[:, 1]
        
        # landmarks = torch.from_numpy(landmarks)
        # landmarks = landmarks.to('cuda')        

        priors = torch.tensor(self.anchors).view(-1, 4)
        priors = priors.to('cuda')

        # pre = landmarks.squeeze(0) 
        pre = torch.squeeze(landmarks, 0)
        
        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)
        # landmarks = landmarks * scale1
        landmarks = torch.mul(landmarks, scale1)

        landmarks = landmarks.cpu().numpy()  

        # ignore low scores
        # inds = np.where(scores > 0.97)[0]
        inds = torch.where(scores>0.97)[0]
        inds = inds.cpu().numpy()  
        scores = scores.cpu().numpy()  
        
        landmarks, scores = landmarks[inds], scores[inds]    

        # sort
        order = scores.argsort()[::-1]
        landmarks = landmarks[order][0]

        return np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])

    def detect(self, img, input_size, max_num=0, metric='default'):
        # Resize image to fit within the input_size
        im_ratio = torch.div(img.size()[1], img.size()[2])

        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device=device)
        det_img[:new_height,:new_width,  :] = img

        # Switch to BGR and normalize
        det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128
        
        # Prepare data and find model parameters 
        det_img = torch.unsqueeze(det_img, 0).contiguous()
        input_name = self.detection_model.get_inputs()[0].name
        
        outputs = self.detection_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        
        io_binding = self.detection_model.io_binding() 
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=det_img.size(), buffer_ptr=det_img.data_ptr())
        
        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda') 
        
        # Sync and run model
        syncvec = self.syncvec.cpu()     
        self.detection_model.run_with_iobinding(io_binding)
        
        net_outs = io_binding.copy_outputs_to_cpu()

        input_height = det_img.shape[2]
        input_width = det_img.shape[3]
        
        fmc = 3
        center_cache = {}
        scores_list = []
        bboxes_list = []
        kpss_list = []
        for idx, stride in enumerate([8, 16, 32]):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride

            kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                if len(center_cache)<100:
                    center_cache[key] = anchor_centers
            
            pos_inds = np.where(scores>=0.5)[0]

            x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
            y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
            x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
            y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

            bboxes = np.stack([x1, y1, x2, y2], axis=-1)  
            
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            preds = []
            for i in range(0, kps_preds.shape[1], 2):
                px = anchor_centers[:, i%2] + kps_preds[:, i]
                py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                preds.append(px)
                preds.append(py)
            kpss = np.stack(preds, axis=-1) 
            #kpss = kps_preds
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        
        det_scale = det_scale.numpy()###
        
        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        
        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]        

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            det_img_center = det_img.shape[0] // 2, det_img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            if kpss is not None:
                kpss = kpss[bindex, :]
                
        return kpss   

    def recognize(self, img, face_kps):
        # Find transform 
        tform = trans.SimilarityTransform()
        tform.estimate(face_kps, self.arcface_dst)

        # Transform
        img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) ) 
        img = v2.functional.crop(img, 0,0, 112, 112)

        # Switch to BGR and normalize
        img = img.permute(1,2,0) #112,112,3        
        img = img[:, :, [2,1,0]]
        img = torch.sub(img, 127.5)
        img = torch.div(img, 127.5)        
        img = img.permute(2, 0, 1) #3,112,112
        
        # Prepare data and find model parameters        
        img = torch.unsqueeze(img, 0).contiguous()     
        input_name = self.recognition_model.get_inputs()[0].name
        
        outputs = self.recognition_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        
        io_binding = self.recognition_model.io_binding() 
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=img.size(), buffer_ptr=img.data_ptr())

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda') 
        
        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.recognition_model.run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten()      
    
    def clear_mem(self):
        del self.swapper_model
        del self.GFPGAN_model
        del self.occluder_model
        del self.face_parsing_model
        del self.codeformer_model
        del self.GPEN_256_model
        del self.GPEN_512_model
        del self.resnet_model
        del self.detection_model
        del self.recognition_model
        
        self.swapper_model = []  
        self.GFPGAN_model = []
        self.occluder_model = []
        self.face_parsing_model = []
        self.codeformer_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.resnet_model = []
        self.detection_model = []
        self.recognition_model = []
                
        # test = swap.permute(1, 2, 0)
        # test = test.cpu().numpy()
        # cv2.imwrite('2.jpg', test) 