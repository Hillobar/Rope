import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from numpy.linalg import norm as l2norm
from skimage import transform as trans
from insightface.utils.face_align import norm_crop2
import subprocess
from math import floor, ceil

import torch
import requests
from PIL import Image
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
        self.occluder_tensor = []        
        self.face_parsing_model = []
        self.face_parsing_tensor = []   
        
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
        self.set_read_threads = []          # Name of threaded function
        self.frame_timer = time.time()      # used to set the framerate during playing
        self.play_frame_tracker = -1        # tracks the next frame during playing in case the threads return out of order
        
        # Queues
        self.action_q = []                  # queue for sending to the coordinator
        self.frame_q = []                   # queue for frames that are ready for coordinator
        self.frame_q2 = []                  # queue for frames created by thread and ready to be added to frame_q
        self.r_frame_q = []                 # queue for frames that are requested by the GUI
        self.read_video_frame_q = []
        
        # swapping related
        self.source_embedding = []          # array with indexed source embeddings
        self.swap = False                   # flag for the swap enabled toggle
        self.found_faces_assignments = []   # array that maps the found faces to source faces    

        self.parameters = []

        self.num_threads = 0
        self.target_video = []

        self.fps = 1.0
        self.temp_file = []

        self.i_image = []
        self.io_binding = False
        self.video_read_success = False
        self.clip_session = []
        self.cuda_device = []

        self.start_time = []
        self.record = False
        self.output = []

        self.saved_video_path = []
        self.sp = []
        self.timer = []

        self.clip_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Resize((256, 256))])

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
        # print(self.fps)

        
        if not self.capture.isOpened():
            print("Cannot open file: ", file)
            exit()
        else:
            self.target_video = file
            self.is_video_loaded = True
            self.video_frame_total = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.play = False 
            self.current_frame = 0
         
            self.set_read_threads = []
            self.frame_timer = time.time()
            self.play_frame_tracker = 0
 
            self.frame_q = []
            self.frame_q2 = []                  
            self.r_frame_q = [] 
            
            self.swap = False 
            self.found_faces_assignments = []

            self.add_action("set_slider_length",self.video_frame_total-1)

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        success, image = self.capture.read()        
        if success:
            crop = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            temp = [crop, 0]
            self.frame_q.append(temp)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

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
            self.play_video(False)
            self.current_frame = int(frame)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, min(self.video_frame_total, self.current_frame))
            success, target_image = self.capture.read()
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, min(self.video_frame_total, self.current_frame))
            if success:
                # target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                if not self.swap:   
                    temp = [target_image, self.current_frame]
                else:  
                    temp = [self.swap_video(target_image), self.current_frame]
                temp[0] = cv2.cvtColor(temp[0], cv2.COLOR_BGR2RGB) 
                self.r_frame_q.append(temp)     
           
    def play_video(self, command):     #"record", "play", "stop"       
        if command == "play":
            self.play = True
            self.play_frame_tracker = self.current_frame          
            # self.capture.set(cv2.CAP_PROP_POS_FRAMES, min(self.video_frame_total, self.current_frame))     

        if command == "stop":
            self.play = False
        if command == "record":
            self.record = True
            self.play = True
            # Initialize
            self.timer = time.time()
            frame_width = int(self.capture.get(3))
            frame_height = int(self.capture.get(4))
            frame_size = (frame_width,frame_height)

            
            self.play_frame_tracker = self.current_frame 
            # self.start_time = self.capture.get(cv2.CAP_PROP_POS_MSEC)/1000.0
            self.start_time = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))
            
            
            self.file_name = os.path.splitext(os.path.basename(self.target_video))

            base_filename =  self.file_name[0]+"_"+str(time.time())[:10]

            self.output = os.path.join(self.saved_video_path, base_filename)

            self.temp_file = self.output+"_temp"+self.file_name[1]  
            
            data = subprocess.run(['ffprobe', '-loglevel', 'error', '-show_streams', '-of', 'json', f'{self.target_video}'], capture_output=True).stdout
            d = json.loads(data)

            # if d['streams'][0]['codec_name'] =='vp9':
                            # args = ["ffmpeg", 
                # "-an",            
                # "-r", str(self.fps),
                # "-i", "pipe:",
                # "-vf", "format="+d['streams'][0]['pix_fmt'],
                # "-vcodec", d['streams'][0]['codec_name'],
                # "-r", str(self.fps),
                # "-s", str(frame_width)+"x"+str(frame_height),
                # final_file] 

            args = ["ffmpeg", 
                "-an",       
                "-r", str(self.fps),
                "-i", "pipe:",
                "-vf", "format=yuvj420p",
                "-c:v", "libx264",
                "-crf", str(self.vid_qual),
                "-r", str(self.fps),
                "-s", str(frame_width)+"x"+str(frame_height),
                self.temp_file]  



            self.sp = subprocess.Popen(args, stdin=subprocess.PIPE)





    def process(self):

        if len(self.set_read_threads) != self.num_threads:
            self.set_read_threads = [[0] * 4 for i in range(self.num_threads)]
        
        # Add threads to Queue
        if self.play == True and self.is_video_loaded == True:
            for i in range(self.num_threads):
                if self.set_read_threads[i][3] == 0:
                    self.set_read_threads [i] = [threading.Thread(target=self.thread_video_read, args = [self.current_frame]).start(), 0, self.current_frame, 1]
                    self.current_frame += 1
                    break

        else:
            self.play == False

        # Always be emptying the queues
        time_diff = time.time() - self.frame_timer

        if not self.record and time_diff >= 1.0/float(self.fps):
            for i in range(self.num_threads):
                if self.set_read_threads[i][3] == 2 and self.set_read_threads[i][2] == self.play_frame_tracker:
                    
                    temp = [self.set_read_threads[i][1], self.set_read_threads[i][2]]
                    self.frame_q.append(temp)
                    fps = round(1.0/time_diff, 1)
                    msg = "Playing at %s fps" %fps
                    self.add_action("send_msg", msg)
                    self.play_frame_tracker += 1
                    self.set_read_threads[i][3] = 0
                    self.frame_timer = time.time()
                    break    
                    
        elif self.record:
            empty_count = 0
            for i in range(self.num_threads):
                # print(self.set_read_threads[i][3])
                if self.set_read_threads[i][3] == 2 and self.set_read_threads[i][2] == self.play_frame_tracker:
                    
                    temp = [self.set_read_threads[i][1], self.set_read_threads[i][2]]
                    self.frame_q.append(temp)

                    image = self.set_read_threads[i][1]
                    pil_image = Image.fromarray(image)
                    pil_image.save(self.sp.stdin, 'JPEG')                    
                    framen = self.play_frame_tracker
                    msg = "Rendering frame %s/%s" %(framen, self.video_frame_total-1)    

                    self.play_frame_tracker += 1
                    self.set_read_threads[i][3] = 0
                    self.frame_timer = time.time()
                    break  
                elif self.set_read_threads[i][3] == 0:
                    empty_count = empty_count + 1

            if empty_count == self.num_threads:
        # Close video and process

                stop_time = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))
                if stop_time == 0:
                    stop_time = float(self.video_frame_total) / float(self.fps)
                
                self.sp.stdin.close()
                self.sp.wait()

                orig_file = self.target_video
                final_file = self.output+self.file_name[1]
                self.add_action("send_msg", "adding audio...")    
                args = ["ffmpeg",
                        "-i", self.temp_file,
                        "-ss", str(self.start_time), "-to", str(stop_time), "-i",  orig_file,
                        "-c",  "copy", # may be c:v
                        "-map", "0:v:0", "-map", "1:a:0?",
                        "-shortest",
                        final_file]
                
                four = subprocess.run(args)

                os.remove(self.temp_file)

                self.record = False
                timef= time.time() - self.timer 
                msg = "done...total rendering time: %s seconds" % round(timef,1)
                self.add_action("send_msg", msg) 

    def thread_video_read(self, frame_number):   
        # frame_timer = time.time()

        with lock:
            success, target_image = self.capture.read()

        if success:
            if not self.swap:
                temp = [target_image, frame_number]
            else:
                temp = [self.swap_video(target_image), frame_number]
            temp[0] = cv2.cvtColor(temp[0], cv2.COLOR_BGR2RGB) 
            for i in range(len(self.set_read_threads)):
                if self.set_read_threads[i][2] == frame_number:
                    self.set_read_threads[i][1] = temp[0]
                    self.set_read_threads[i][3] = 2
                    break
      
        else:
            for i in range(len(self.set_read_threads)):
                if self.set_read_threads[i][2] == frame_number:
                    self.set_read_threads[i][3] = 0
                    break

            self.play = False
            self.add_action("stop_play", True)
        # time_diff = time.time() - frame_timer
        # print( time_diff) 

    def load_source_embeddings(self, source_embeddings):
        self.source_embedding = []
        for i in range(len(source_embeddings)):
            self.source_embedding.append(source_embeddings[i]["Embedding"])
    
    def swap_set(self, swap):
        self.swap = swap
        # self.get_video_frame(self.current_frame)
     
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
            
 
        
    def set_faceapp_model(self, faceapp):
        self.faceapp_model = faceapp

    def swap_video(self, target_image):        
        # Find faces, returns all faces
        ret = self.faceapp_model.get(target_image, max_num=10)
        if ret:
            img = target_image
            target_face = ret

            # Loop through target faces to see if they match our target embeddings
            for i in range(len(target_face)):
                for j in range(len(self.found_faces_assignments)):
                    # sim between face in video and already found face
                    sim = self.findCosineDistance(target_face[i].embedding, self.found_faces_assignments[j]["Embedding"])
                  
                    # if the face[i] in the frame matches afound face[j] AND the found face is active (not []) 
                    if self.parameters["ThreshholdState"]:
                        threshhold = 2.0
                    else:    
                        threshhold = self.parameters["Threshhold"]
                    
                    if sim<float(threshhold) and self.found_faces_assignments[j]["SourceFaceAssignments"]:
                        total_s_e = self.source_embedding[self.found_faces_assignments[j]["SourceFaceAssignments"][0]]   
                        for k in range(1,len(self.found_faces_assignments[j]["SourceFaceAssignments"])):
                            total_s_e = total_s_e + self.source_embedding[self.found_faces_assignments[j]["SourceFaceAssignments"][k]]

                        s_e =  total_s_e / len(self.found_faces_assignments[j]["SourceFaceAssignments"])
                        img = self.swap_core(img, target_face[i].kps, s_e, target_face[i].bbox) 

            return img
        else:
            return target_image
            
        # self.target_face = {    
                                # "TKButton":                 [],
                                # "ButtonState":              "off",
                                # "Image":                    [],
                                # "Embedding":                [],
                                # "SourceFaceAssignments":    []
                            # }              
            
            
            
       
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
    def swap_core(self, img, kps,  s_e, bbox):

        # 128 transforms
        ratio = 1.0
        diff_x = 8.0*ratio
        dst = self.arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst)
        M128 = tform.params[0:2, :]    

        # 512 transforms
        ratio = 4.0
        diff_x = 8.0*ratio
        dst = self.arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst)
        M512 = tform.params[0:2, :]
        IM512 = cv2.invertAffineTransform(M512)
        
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

        original_face_512 = cv2.warpAffine(img, M512, (512,512), borderValue=0.0)
        original_face_256 = cv2.resize(original_face_512, (256,256))
        original_face = cv2.resize(original_face_256, (128, 128))
        # cv2.imwrite('./original_face_512.jpg',original_face_512 )

        blob = cv2.dnn.blobFromImage(original_face, 1.0 / 255.0, self.input_size, (0.0, 0.0, 0.0), swapRB=True)

        #Select source embedding
        n_e = s_e / l2norm(s_e)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        
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


        img_fake = pred.transpose((0,2,3,1))[0]
        swapped_face = np.clip(255 * img_fake, 0, 255).astype(np.float32)[:,:,::-1]
        swapped_face_upscaled = cv2.resize(swapped_face, (512,512))

        border_mask = np.zeros((128, 128), dtype=np.float32)        
        border_mask = cv2.rectangle(border_mask, (int(self.parameters["MaskSide"]), int(self.parameters["MaskTop"])), (128-int(self.parameters["MaskSide"]), 128-5), (255, 255, 255), -1)/255.0    
        border_mask = cv2.GaussianBlur(border_mask, (self.parameters["MaskBlur"]*2+1,self.parameters["MaskBlur"]*2+1),0)
        
        img_mask = np.ones((128, 128), dtype=np.float32)  
        
        if self.parameters["GFPGANState"]:           
            swapped_face_upscaled = self.apply_GFPGAN(swapped_face_upscaled) 

        # Occluder
        if self.parameters["OccluderState"]:
            occlude_mask = self.apply_occlusion(original_face_256)
            occlude_mask = cv2.resize(occlude_mask, (128,128))  
            img_mask *= occlude_mask 
            
            # parse_mask = 1.0-self.apply_face_parser_nose(swapped_face_upscaled)
            # img_mask += parse_mask
            # img_mask = np.clip(img_mask, 0, 1)

            # cv2.imwrite('occluder.png', img_mask*255) 

        # CLIPs CLIPs
        if self.parameters["CLIPState"]:
            mask = self.apply_neg_CLIPs(original_face_512)
            mask = cv2.resize(mask, (128,128))
            img_mask *= mask

        # Face Parsing
        if self.parameters["FaceParserState"]:
            mask = self.apply_face_parser(swapped_face_upscaled)
            mask *= self.apply_face_parser(original_face_512)
            mask = cv2.resize(mask, (128,128))
            img_mask *= mask

        # Face Diffing
        if self.parameters["DiffState"]:
            fake_diff = self.apply_fake_diff(swapped_face, original_face)
            fake_diff /= 255
            img_mask *= fake_diff
        
        img_mask = cv2.GaussianBlur(img_mask, (self.parameters["BlurAmount"]*2+1,self.parameters["BlurAmount"]*2+1),0)
        img_mask *= border_mask
          

        img_mask = cv2.resize(img_mask, (512,512))
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1]) 
        # print(swapped_face_upscaled.dtype, img_mask.dtype)
        swapped_face_upscaled *= img_mask

        swapped_face_upscaled = cv2.warpAffine(swapped_face_upscaled, IM512, (img.shape[1], img.shape[0]), borderValue=0.0) 
     
        # img_mask = cv2.warpAffine(img_mask, IM512, (img.shape[1], img.shape[0]), borderValue=0.0)
        # img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])        
        # index = np.where(img_mask != 0)        
        # img[index[0], index[1]] = (1.0-img_mask[index[0], index[1]])*img[index[0], index[1]] + swapped_face_upscaled[index[0], index[1]]
        
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
      
        swapped_face_upscaled += img_mask*img_a
        
        img[top:bottom, left:right, 0:3] = swapped_face_upscaled        
 

        return img.astype(np.uint8)   #BGR
        


 
    def apply_occlusion(self, img):
        data = self.occluder_tensor(img).unsqueeze(0)
        data = data.to('cuda')
        with lock:
            with torch.no_grad():
                pred = self.occluder_model(data)
        occlude_mask = (pred > 0).type(torch.float32)
        occlude_mask = occlude_mask.squeeze().cpu().numpy()*1.0
 
        return occlude_mask 
    
      
    def apply_neg_CLIPs(self, img):
        clip_mask = np.ones((256, 256))
        CLIPimg = self.clip_transform(img).unsqueeze(0)

        if self.parameters["CLIPText"] != "":
            prompts = self.parameters["CLIPText"].split(',')
            
            with lock:
                with torch.no_grad():
                    preds = self.clip_session(CLIPimg.repeat(len(prompts),1,1,1), prompts)[0]

            clip_mask = 1 - torch.sigmoid(preds[0][0])
            for i in range(len(prompts)-1):
                clip_mask *= 1-torch.sigmoid(preds[i+1][0])
            clip_mask = clip_mask.data.cpu().numpy()
           
            clip_mask[clip_mask>self.parameters["CLIPAmount"]] = 1.0
            clip_mask[clip_mask<=self.parameters["CLIPAmount"]] = 0.0

        return clip_mask     

    def apply_face_parser(self, img):

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

        
        with lock:
            with torch.no_grad():
                img1 = self.face_parsing_tensor(img.astype(np.uint8))
                img1 = torch.unsqueeze(img1, 0)
                img1 = img1.cuda()
                out = self.face_parsing_model(img1)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)

        vis_parsing_anno = parsing.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.ones((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))

        index = np.where((vis_parsing_anno == 11) | (vis_parsing_anno == 12) | (vis_parsing_anno == 13))
        # index = np.where(vis_parsing_anno == 11)
        vis_parsing_anno_color[index[0], index[1]] = 0.0
        # kernel = np.ones((2, 2))
        # vis_parsing_anno_color = cv2.erode(vis_parsing_anno_color, kernel, iterations=10)
        return vis_parsing_anno_color   

    def apply_face_parser_nose(self, img):

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

        
        with lock:
            with torch.no_grad():
                img1 = self.face_parsing_tensor(img)
                img1 = torch.unsqueeze(img1, 0)
                img1 = img1.cuda()
                out = self.face_parsing_model(img1)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)

        vis_parsing_anno = parsing.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.ones((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))

        index = np.where((vis_parsing_anno == 10) | (vis_parsing_anno == 17))
        vis_parsing_anno_color[index[0], index[1]] = 0.0
        
        return vis_parsing_anno_color         
        
    def apply_GFPGAN(self, swapped_face_upscaled):
        

        temp = swapped_face_upscaled

        # preprocess
        # temp = cv2.resize(temp, (512, 512))
        temp = temp / 255.0
        # temp = temp.astype('float32')
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        temp[:,:,0] = (temp[:,:,0]-0.5)/0.5
        temp[:,:,1] = (temp[:,:,1]-0.5)/0.5
        temp[:,:,2] = (temp[:,:,2]-0.5)/0.5
        temp = np.float32(temp[np.newaxis,:,:,:])
        temp = temp.transpose(0, 3, 1, 2)

        ort_inputs = {"input": temp}
        if self.io_binding:
            io_binding = self.GFPGAN_model.io_binding()            
            io_binding.bind_cpu_input("input", temp)
            io_binding.bind_output("1288", "cuda")
               
            self.GFPGAN_model.run_with_iobinding(io_binding)
            ort_outs = io_binding.copy_outputs_to_cpu()
        else:
            
            ort_outs = self.GFPGAN_model.run(None, ort_inputs)
        
        output = ort_outs[0][0]

        # postprocess
        output = output.clip(-1,1)
        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (output * 255.0).round()





        temp2 = float(self.parameters["GFPGANAmount"])/100.0
        swapped_face_upscaled = cv2.addWeighted(output, temp2, swapped_face_upscaled, 1.0-temp2,0)
        
        return swapped_face_upscaled
        
    def apply_fake_diff(self, swapped_face, original_face):
        fake_diff = swapped_face.astype(np.float32) - original_face.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2,:] = 0
        fake_diff[-2:,:] = 0
        fake_diff[:,:2] = 0
        fake_diff[:,-2:] = 0        
        
        fthresh = int(self.parameters["DiffAmount"])
        fake_diff[fake_diff<fthresh] = 0
        fake_diff[fake_diff>=fthresh] = 255 

        return fake_diff    
            



    # @profile    
