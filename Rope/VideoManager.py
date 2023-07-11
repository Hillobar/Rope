import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from numpy.linalg import norm as l2norm
from skimage import transform as trans
import insightface
import subprocess

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
        self.fps = []
        
        self.i_image = []
  
    def load_target_video( self, file ):
        # If we already have a video loaded, release it
        if self.capture:
            self.capture.release()
            
        # Open file                
        self.capture = cv2.VideoCapture(file)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

        
        if not self.capture.isOpened():
            print("Cannot open file: ", file)
            exit()
        else:
            self.target_video = file
            self.is_video_loaded = True
            self.video_frame_total = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)-1)
            self.play = False 
            self.current_frame = 0
         
            self.set_read_threads = []
            self.frame_timer = time.time()
            self.play_frame_tracker = -1
 
            self.frame_q = []
            self.frame_q2 = []                  
            self.r_frame_q = [] 
            
            self.swap = False 
            self.found_faces_assignments = []

            self.add_action("set_slider_length",self.video_frame_total)

        success, image = self.capture.read()
        if success:
            crop = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            temp = [crop, 0]
            self.r_frame_q.append(temp) 

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
     
 ##############################   
    def get_frame(self):
        # temp_return = False
        # for i in range(len(self.frame_q)):
            # if self.frame_q[i][1] == self.play_frame_tracker:
                               
                # frame = self.frame_q[i]
                # self.frame_q.pop(i)
                # self.play_frame_tracker += 1
                # return frame
        # return temp_return
        frame = self.frame_q[0]
        self.frame_q.pop(0)
        return frame
    
    def get_frame_length(self):
        return len(self.frame_q)  
        
    def get_requested_frame(self):
        frame = self.r_frame_q[0]
        self.r_frame_q.pop(0)
        #self.set_read_threads.pop(0)
        #print(len(self.set_read_threads))
        return frame
    
    def get_requested_frame_length(self):
        return len(self.r_frame_q)          
###################################    

    def get_video_frame(self, frame):    
        if self.is_video_loaded == True:
            self.current_frame = int(frame)
            image = self.capture.set(cv2.CAP_PROP_POS_FRAMES, min(self.video_frame_total, self.current_frame))
            #success, image = self.capture.read()
            if success:
                target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                if not self.swap:   
                    temp = [target_image, self.current_frame]
                    self.frame_q.append(temp)
            
                else:  
                    temp = [self.swap_video(target_image), self.current_frame]
                    self.frame_q.append(temp)      

    def get_requested_video_frame(self, frame):    
        if self.is_video_loaded == True:
            self.play_video(False)
            self.current_frame = int(frame)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, min(self.video_frame_total, self.current_frame))
            success, image = self.capture.read()
            if success:
                target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                if not self.swap:   
                    temp = [target_image, self.current_frame]
                    self.r_frame_q.append(temp)
            
                else:  
                    temp = [self.swap_video(target_image), self.current_frame]
                    self.r_frame_q.append(temp)     
                
    def play_video(self, is_play):            
        self.play = is_play
        if self.play == True:
            self.play_frame_tracker = self.current_frame+1            
            
    def process(self):
        global output
        # Queue frame processing threads
        
        if self.play == True and self.is_video_loaded == True and self.current_frame <= self.video_frame_total-1:
              
            if len(self.frame_q2) < self.num_threads and len(self.set_read_threads) < self.num_threads:
                self.current_frame += 1
                # Set track start to first played frame

                temp = threading.Thread(target=self.thread_video_read, args = [self.current_frame])
                temp.start()
                self.set_read_threads.append(temp)
        else:
            self.play == False

        if self.create_video == False:
            time_diff = time.time() - self.frame_timer
            if len(self.frame_q2) > 0 and time_diff >= 1/float(self.fps):
                for i in range(len(self.frame_q2)):
                    if self.frame_q2[i][1] == self.play_frame_tracker:
                        # print( 1/float(time_diff))
                        self.frame_q.append(self.frame_q2[i])
                        self.frame_q2.pop(i)
                        self.play_frame_tracker += 1
                        self.set_read_threads[i].join()
                        self.set_read_threads.pop(i)
                        self.frame_timer = time.time()
                        break
        else:

            if not self.output_video:
                frame_width = int(self.capture.get(3))
                frame_height = int(self.capture.get(4))
                frame_size = (frame_width,frame_height)
                fps = self.capture.get(cv2.CAP_PROP_FPS)
                
                self.current_frame = -1
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.play_frame_tracker = 0
                self.play = True
                # output = os.path.basename(self.target_video)
                # output = output[:20]+"_"+str(time.time())[:10]
                output = str(time.time())[:10]
                self.output_video = cv2.VideoWriter(output+".mp4", cv2.VideoWriter_fourcc(*'avc1'), fps, frame_size)
               
            elif len(self.frame_q2) > 0 and self.output_video:
                for i in range(len(self.frame_q2)):
                    if self.frame_q2[i][1] == self.play_frame_tracker:
                        # print( time_diff)
                        self.frame_q.append(self.frame_q2[i])
                        image = cv2.cvtColor(self.frame_q2[i][0], cv2.COLOR_BGR2RGB)
                        self.frame_q2.pop(i)
                        self.play_frame_tracker += 1
                        self.set_read_threads[i].join()
                        self.set_read_threads.pop(i)
                        
                        self.output_video.write(image)
                        

                        break
            if self.play_frame_tracker == self.video_frame_total+1 and len(self.frame_q2) == 0:
                self.output_video.release() 
                self.output_video = []
                
                roped_file = output+'.mp4'
                orig_file = self.target_video
                final_file = output+'_a.mp4'
  
                # os.system(f'ffmpeg -i '+roped_file+' -i '+orig_file+' -c copy -map 0:v:0 -map 1:a:0 -shortest '+final_file)
                
                args = ["ffmpeg",
                        "-i",  roped_file,
                        "-i",  orig_file,
                        "-c",  "copy",
                        "-map", "0:v:0", "-map", "1:a:0?",
                        "-shortest",
                        final_file]
                
                four = subprocess.run(args)
                
                final_file_small = output+'_a_s.mp4'
                
                args = ["ffmpeg",
                "-i", final_file,
                "-vcodec", "libx265",
                "-crf", "28",
                final_file_small]
                
                four = subprocess.run(args)
                
                os.remove(roped_file)
                os.remove(final_file)

                self.create_video = False
                self.play == False

    def thread_video_read(self, frame_number):   
        #frame_timer = time.time()
        
        lock.acquire()
        success, image = self.capture.read()
        lock.release()
        if success:
            target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            if self.swap == False:
                temp = [target_image, frame_number]
                self.frame_q2.append(temp)
            else:
                temp = [self.swap_video(target_image), frame_number]
                self.frame_q2.append(temp)
       
        #time_diff = time.time() - frame_timer
        #print( time_diff) 

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
                        img = self.swap_core(img, target_face[i].kps, s_e)            
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
    def swap_core(self, img, kps,  s_e):
        paste_back = True
       
        aimg, _ = insightface.utils.face_align.norm_crop2(img, kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, self.input_size, (0.0, 0.0, 0.0), swapRB=True)
       
       #Select source embedding
        n_e = s_e / l2norm(s_e)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.swapper_model.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
        #print(latent.shape, latent.dtype, pred.shape)
        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        if not paste_back:
            return bgr_fake
        else:
            target_img = img

            ratio = 2.0
            diff_x = 8.0*ratio
            arcface_dst = np.array(
                [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                 [41.5493, 92.3655], [70.7299, 92.2041]],
                dtype=np.float32) 
            dst = arcface_dst * ratio
            dst[:,0] += diff_x
            tform = trans.SimilarityTransform()
            tform.estimate(kps, dst)
            M1 = tform.params[0:2, :]
            IM = cv2.invertAffineTransform(M1)

            bgr_fake_upscaled = cv2.resize(bgr_fake, (256,256))
            
            img_white = np.full((bgr_fake_upscaled.shape[0],bgr_fake_upscaled.shape[1]), 255, dtype=np.float32)
            img_black = np.full((bgr_fake_upscaled.shape[0],bgr_fake_upscaled.shape[1]), 0, dtype=np.float32)
  
            img_white[img_white>20] = 255
            img_mask = img_black
            mask_border = 5
            img_mask = cv2.rectangle(img_mask, (mask_border+int(self.mask_left), mask_border+int(self.mask_top)), 
                                    (256 - mask_border-int(self.mask_right), 256-mask_border-int(self.mask_bottom)), (255, 255, 255), -1)    
            img_mask = cv2.GaussianBlur(img_mask, (self.mask_blur*2+1,self.mask_blur*2+1), 0)    
            img_mask /= 255

            if not self.fake_diff_state:
                img_mask_0 = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            else:
                fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
                fake_diff = np.abs(fake_diff).mean(axis=2)
                fake_diff[:2,:] = 0
                fake_diff[-2:,:] = 0
                fake_diff[:,:2] = 0
                fake_diff[:,-2:] = 0
                fake_diff = cv2.resize(fake_diff, (256,256))
                
                fthresh = int(self.fake_diff_blend)
                fake_diff[fake_diff<fthresh] = 0
                fake_diff[fake_diff>=fthresh] = 255 
                k = 5
                kernel_size = (k, k)
                blur_size = tuple(2*i+1 for i in kernel_size)
                fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
                fake_diff /= 255
                
                img_mask_1 = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
                img_mask_0 = np.reshape(fake_diff, [fake_diff.shape[0],fake_diff.shape[1],1]) 
                
                img_mask_0 *= img_mask_1

            if self.GFPGAN_state:        
                temp, _ = self.GFPGAN_model.forward(bgr_fake_upscaled)
                temp2 = float(self.GFPGAN_blend)/100.0
                bgr_fake_upscaled = cv2.addWeighted(temp, temp2, bgr_fake_upscaled, 1.0-temp2,0)
                #crop = cv2.cvtColor(bgr_fake_upscaled, cv2.COLOR_RGB2BGR) 
                #cv2.imwrite("test.jpg", crop)
            
            fake_merged = img_mask_0* bgr_fake_upscaled
         
            ####   
            if not self.fake_diff_state:
                img_mask = cv2.warpAffine(img_mask, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)

            else:
                img_mask = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
                
            img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])    
            fake_merged = cv2.warpAffine(fake_merged, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0) 
            fake_merged = fake_merged + (1-img_mask) * target_img.astype(np.float32)
            fake_merged = fake_merged.astype(np.uint8)   
            # cv2.imwrite("test.jpg", fake_merged)
        return fake_merged    
        
    def set_GFPGAN_model(self, model):
        self.GFPGAN_model = model
        
    def toggle_GFPGAN(self, state):
        self.GFPGAN_state = state
        
    def toggle_fake_diff(self, state):
        self.fake_diff_state = state
        
