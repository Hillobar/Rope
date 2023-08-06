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
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
import json

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
        self.arcface_dst = []               # a constant used for arcfacing
        
        #Video related
        self.capture = []                   # cv2 video
        self.is_video_loaded = False        # flag for video loaded state    
        self.video_frame_total = None       # length of currently loaded video
        self.play = False                   # flag for the play button toggle
        self.current_frame = 0              # the current frame of the video
        
        
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

        self.fake_diff_blend = 0                        # test var
        self.mask_top = 0
        self.mask_bottom = 0
        self.mask_left = 0
        self.mask_right = 0
        self.mask_blur = 0
        self.GFPGAN_model = []
        self.GFPGAN_state = False
        self.fake_diff_state = False
        self.GFPGAN_blend = 100
        self.create_video = False
        self.output_video = []
        self.num_threads = []
        self.target_video = []
        self.write_threads = []
        self.write_threads_tracker = []
        self.face_thresh = []
        self.fps = 1.0
        self.temp_file = []
        self.file_name = []
        
        self.i_image = []
        self.io_binding = True
        self.video_read_success = False
        self.clip_session = []
        self.cuda_device = []
        self.CLIPs = ['', '']
        self.toggle_CLIPs = []
        self.pos_thresh = []
        self.neg_thresh = []
        self.start_time = []
        self.record = False
        self.output = []
        self.CLIP_blur = []
        self.saved_video_path = []
        self.sp = []
        self.timer = []
        self.occluder_model = []
        self.occluder_tensor = []
        self.occluder = []
        self.occluder_blur = []
        self.vid_qual = []



  
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
###################################    

    # def get_video_frame(self, frame):    
        # if self.is_video_loaded == True:
            # self.current_frame = int(frame)
            # image = self.capture.set(cv2.CAP_PROP_POS_FRAMES, min(self.video_frame_total, self.current_frame))
            # #success, image = self.capture.read()
            # if success:
                # target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                # if not self.swap:   
                    # temp = [target_image, self.current_frame]
                    # self.frame_q.append(temp)
            
                # else:  
                    # temp = [self.swap_video(target_image), self.current_frame]
                    # self.frame_q.append(temp)      

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
            self.start_time = self.capture.get(cv2.CAP_PROP_POS_MSEC)/1000.0
            
            
            
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

        # Add threads to Queue
        if self.play == True and self.is_video_loaded == True:

            if len(self.set_read_threads) < self.num_threads:
                temp = threading.Thread(target=self.thread_video_read, args = [self.current_frame])
                temp.start()                
                self.set_read_threads.append(temp)
                self.current_frame += 1
                
        else:
            self.play == False
            
        if not self.record:
            # Always be emptying the queues
            time_diff = time.time() - self.frame_timer
            if len(self.frame_q2) > 0 and time_diff >= 1.0/float(self.fps):
                for i in range(len(self.frame_q2)):
                    if self.frame_q2[i][1] == self.play_frame_tracker:
                        # print( 1/float(time_diff))
                        self.frame_q.append(self.frame_q2[i])
                        self.frame_q2.pop(i)
                        self.play_frame_tracker += 1
                        self.set_read_threads[i].join()
                        self.set_read_threads.pop(i)
                        fps = round(1.0/time_diff, 1)
                        msg = "Playing at %s fps" %fps
                        self.add_action("send_msg", msg)
                        self.frame_timer = time.time()
                        break    
        
        # Create Video instead
        else:
            # Play through frames in video     
            # if len(self.frame_q2) > 0 and self.output_video:

            if len(self.frame_q2) > 0:
                for i in range(len(self.frame_q2)):
                    if self.frame_q2[i][1] == self.play_frame_tracker:
                        self.frame_q.append(self.frame_q2[i])
                        image = self.frame_q2[i][0]
                        self.frame_q2.pop(i)
                        framen = self.play_frame_tracker
                        msg = "Rendering frame %s/%s" %(framen, self.video_frame_total-1)
                        self.play_frame_tracker += 1
                        self.set_read_threads[i].join()
                        self.set_read_threads.pop(i)

                        pil_image = Image.fromarray(image)
                        pil_image.save(self.sp.stdin, 'JPEG')
                        
                        self.add_action("send_msg", msg)

                        break
            
            # Close video and process
            elif self.play == False and len(self.frame_q2) == 0:
                stop_time = self.capture.get(cv2.CAP_PROP_POS_MSEC)/1000.0
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
            # target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            if not self.swap:
                temp = [target_image, frame_number]
            else:
                temp = [self.swap_video(target_image), frame_number]
            temp[0] = cv2.cvtColor(temp[0], cv2.COLOR_BGR2RGB) 
            self.frame_q2.append(temp)   
            
            # End gracefully
            if frame_number == self.video_frame_total-1:
                self.play = False
                self.add_action("stop_play", True)
                
        else:
            self.play = False
            self.add_action("stop_play", True)
        # time_diff = time.time() - frame_timer
        # print( time_diff) 

    def load_source_embeddings(self, source_embeddings):
        self.source_embedding = []
        for i in range(len(source_embeddings)):
            self.source_embedding.append(source_embeddings[i][1])
    
    def load_found_faces_assignments(self, found_faces_assignments):
        self.found_faces_assignments = found_faces_assignments
        #self.get_video_frame(self.current_frame) 
    
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
            
        self.arcface_dst = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)    
        
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
                    sim = self.findCosineDistance(target_face[i].embedding, self.found_faces_assignments[j][0])
                    # if the face[i] in the frame matches afound face[j] AND the found face is active (not -1) 
                    if sim<float(self.face_thresh) and self.found_faces_assignments[j][1] != -1:                        
                        s_e =  self.source_embedding[self.found_faces_assignments[j][1]]
                        img = self.swap_core(img, target_face[i].kps, s_e, target_face[i].bbox)  

            return img
        else:
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

    #@profile    
    def swap_core(self, img, kps,  s_e, bbox):

        aimg, _ = norm_crop2(img, kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, self.input_size, (0.0, 0.0, 0.0), swapRB=True)

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
        #print(latent.shape, latent.dtype, pred.shape)
        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        bb = cv2.cvtColor(bgr_fake, cv2.COLOR_BGR2RGB)


        target_img = img

        ratio = 4.0
        diff_x = 8.0*ratio

        dst = self.arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst)
        M1 = tform.params[0:2, :]
        IM = cv2.invertAffineTransform(M1)
        
        ratio = 2.0
        diff_x = 8.0*ratio

        dst = self.arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst)
        M2 = tform.params[0:2, :]


        bgr_fake_upscaled = cv2.resize(bgr_fake, (512,512))
        

        img_white = np.full((bgr_fake_upscaled.shape[0],bgr_fake_upscaled.shape[1]), 255, dtype=np.float32)
        img_black = np.full((bgr_fake_upscaled.shape[0],bgr_fake_upscaled.shape[1]), 0, dtype=np.float32)

        img_white[img_white>20] = 255
        img_mask = img_black
        mask_border = 5
        img_mask = cv2.rectangle(img_mask, (mask_border+int(self.mask_left), mask_border+int(self.mask_top)), 
                                (512 - mask_border-int(self.mask_right), 512-mask_border-int(self.mask_bottom)), (255, 255, 255), -1)    
        img_mask = cv2.GaussianBlur(img_mask, (self.mask_blur*2+1,self.mask_blur*2+1), 0)    
        img_mask /= 255
        
        # Occluder
        if self.occluder:

            input_image = cv2.warpAffine(target_img, M2, (256, 256), borderValue=0.0)

            data = self.occluder_tensor(input_image).unsqueeze(0)

            data = data.to('cuda')
            with lock:
                with torch.no_grad():
                    pred = self.occluder_model(data)

            occlude_mask = (pred > 0).type(torch.int8)

            occlude_mask = occlude_mask.squeeze().cpu().numpy()*1.0



            occlude_mask = cv2.GaussianBlur(occlude_mask*255, (self.occluder_blur*2+1,self.occluder_blur*2+1), 0)
            
            occlude_mask = cv2.resize(occlude_mask, (512,512))
            
            occlude_mask /= 255
            img_mask *= occlude_mask 

            
       
        if self.toggle_CLIPs:
            clip_mask = img_white
            input_image = cv2.warpAffine(target_img, M1, (512, 512), borderValue=0.0)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((512, 512)),
            ])
            img = transform(input_image).unsqueeze(0)

            if self.CLIPs[0] != "":
                prompts = self.CLIPs[0].split(',')
                
                lock.acquire()
                with torch.no_grad():
                    preds = self.clip_session(img.repeat(len(prompts),1,1,1), prompts)[0]
                lock.release()
                clip_mask = torch.sigmoid(preds[0][0])
                for i in range(len(prompts)-1):
                    clip_mask += torch.sigmoid(preds[i+1][0])
                clip_mask = clip_mask.data.cpu().numpy()
                np.clip(clip_mask, 0, 1)
                
                clip_mask[clip_mask>self.pos_thresh] = 1.0
                clip_mask[clip_mask<=self.pos_thresh] = 0.0
                kernel = np.ones((5, 5), np.float32)
                clip_mask = cv2.dilate(clip_mask, kernel, iterations=1)
                clip_mask = cv2.GaussianBlur(clip_mask, (self.CLIP_blur*2+1,self.CLIP_blur*2+1), 0)
                
                img_mask *= clip_mask
            
            
            if self.CLIPs[1] != "":
                prompts = self.CLIPs[1].split(',')
                
                lock.acquire()
                with torch.no_grad():
                    preds = self.clip_session(img.repeat(len(prompts),1,1,1), prompts)[0]
                lock.release()
                neg_clip_mask = torch.sigmoid(preds[0][0])
                for i in range(len(prompts)-1):
                    neg_clip_mask += torch.sigmoid(preds[i+1][0])
                neg_clip_mask = neg_clip_mask.data.cpu().numpy()
                np.clip(neg_clip_mask, 0, 1)

                neg_clip_mask[neg_clip_mask>self.neg_thresh] = 1.0
                neg_clip_mask[neg_clip_mask<=self.neg_thresh] = 0.0
                kernel = np.ones((5, 5), np.float32)
                neg_clip_mask = cv2.dilate(neg_clip_mask, kernel, iterations=1)
                neg_clip_mask = cv2.GaussianBlur(neg_clip_mask, (self.CLIP_blur*2+1,self.CLIP_blur*2+1), 0) 
            
                img_mask -= neg_clip_mask
                # np.clip(img_mask, 0, 1)
                img_mask[img_mask<0.0] = 0.0

        if not self.fake_diff_state:
            img_mask_0 = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            
            img_mask = cv2.warpAffine(img_mask, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        else:
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2,:] = 0
            fake_diff[-2:,:] = 0
            fake_diff[:,:2] = 0
            fake_diff[:,-2:] = 0
            fake_diff = cv2.resize(fake_diff, (512,512))
            
            fthresh = int(self.fake_diff_blend)
            fake_diff[fake_diff<fthresh] = 0
            fake_diff[fake_diff>=fthresh] = 255 

            fake_diff = cv2.GaussianBlur(fake_diff, (15*2+1,15*2+1), 0)

            fake_diff /= 255
            
            img_mask_1 = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            img_mask_0 = np.reshape(fake_diff, [fake_diff.shape[0],fake_diff.shape[1],1]) 
            
            img_mask_0 *= img_mask_1
            
            img_mask = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)

        if self.GFPGAN_state:  

            temp = bgr_fake_upscaled
            # height, width = temp.shape[0], temp.shape[1]

            # preprocess
            temp = cv2.resize(temp, (512, 512))
            temp = temp / 255.0
            temp = temp.astype('float32')
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


            inv_soft_mask = np.ones((512, 512, 1), dtype=np.float32)
            output = cv2.resize(output, (512, 512))
     
            output = output.astype(np.uint8)

            temp2 = float(self.GFPGAN_blend)/100.0
            bgr_fake_upscaled = cv2.addWeighted(output, temp2, bgr_fake_upscaled, 1.0-temp2,0)
            #crop = cv2.cvtColor(bgr_fake_upscaled, cv2.COLOR_RGB2BGR) 
            #cv2.imwrite("test.jpg", crop)
        
        fake_merged = img_mask_0* bgr_fake_upscaled
        
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])    
        fake_merged = cv2.warpAffine(fake_merged, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0) 
       
        diff_hor = (bbox[2]-bbox[0])*0.3
        diff_vert = (bbox[3]-bbox[1])*0.3
        
        left = floor(bbox[0]-diff_hor)
        if left<0:
            left=0
        top = floor(bbox[1]-diff_vert)
        if top<0: 
            top=0
        right = ceil(bbox[2]+diff_hor)
        if right>target_img.shape[1]:
            right=target_img.shape[1]
        
        bottom = ceil(bbox[3]+diff_vert)
        if bottom>target_img.shape[0]:
            bottom=target_img.shape[0]
        
        fake_merged = fake_merged[top:bottom, left:right, 0:3]
        target_img_a = target_img[top:bottom, left:right, 0:3]
        img_mask = img_mask[top:bottom, left:right, 0:1]
        
        fake_merged = fake_merged + (1-img_mask) * target_img_a.astype(np.float32)
        
        target_img[top:bottom, left:right, 0:3] = fake_merged
        
        fake_merged = target_img.astype(np.uint8)   
        
        # fake_merged = fake_merged + (1-img_mask) * target_img.astype(np.float32)
        # fake_merged = fake_merged.astype(np.uint8)   
            
        return fake_merged    #BGR
        
    def set_GFPGAN_model(self, model):
        self.GFPGAN_model = model
        
    def toggle_GFPGAN(self, state):
        self.GFPGAN_state = state
        
    def toggle_fake_diff(self, state):
        self.fake_diff_state = state
     
 
