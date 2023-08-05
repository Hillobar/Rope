# #!/usr/bin/env python3
import os
import time

import rope.GUI as GUI
import rope.VideoManager as VM

from insightface.app import FaceAnalysis
import onnxruntime
import onnx

import torch
from models.clipseg import CLIPDensePredT

import segmentation_models_pytorch as smp
from collections import OrderedDict
from torchvision import transforms



def coordinator():
    global gui, vm, faceapp, swapper, action, frame, r_frame, GFPGAN_session
    start = time.time()
    
    if gui.get_action_length() > 0:
        action.append(gui.get_action())
    if vm.get_action_length() > 0:
        action.append(vm.get_action())
##################    
    if vm.get_frame_length() > 0:
        frame.append(vm.get_frame())
        
    if len(frame) > 0:
        gui.set_image(frame[0], False)
        gui.display_image_in_video_frame()
        frame.pop(0)
 ####################   
    if vm.get_requested_frame_length() > 0:
        r_frame.append(vm.get_requested_frame())    
    if len(r_frame) > 0:
        # print ("1:", time.time())
        gui.set_image(r_frame[0], True)
        gui.display_image_in_video_frame()
        r_frame=[]
 ####################   
    if len(action) > 0:
        if action[0][0] == "load_target_video":
            vm.load_target_video(action[0][1])
            #gui.set_slider_position(0)
            action.pop(0)
        elif action[0][0] == "play_video":
            vm.play_video(action[0][1])
            action.pop(0)
        elif action[0][0] == "set_video_position":
            vm.get_requested_video_frame(action[0][1])
            action.pop(0)
        elif action[0][0] == "find_faces":
            gui.find_faces(action[0][1])
            action.pop(0)    
        elif action[0][0] == "clear_faces":
            gui.clear_faces()
            action.pop(0)    
        elif action[0][0] == "swap":
            vm.swap_set(action[0][1])
            action.pop(0)
        elif action[0][0] == "source_embeddings":  
            vm.load_source_embeddings(action[0][1])
            action.pop(0)
        elif action[0][0] == "found_assignments":
            vm.load_found_faces_assignments(action[0][1])
            action.pop(0)
        elif action[0][0] == "variable":
            vm.set_var(action[0][1])
            action.pop(0)
        elif action[0][0] == "gfpgan_checkbox":
            vm.toggle_GFPGAN(action[0][1])
            action.pop(0)
        elif action[0][0] == "fake_diff_checkbox":
            vm.toggle_fake_diff(action[0][1])
            action.pop(0)
        elif action[0][0] == "top_blend":
            vm.mask_top = action[0][1]
            action.pop(0)
        elif action[0][0] == "bottom_blend":
            vm.mask_bottom = action[0][1]
            action.pop(0)
        elif action[0][0] == "left_blend":
            vm.mask_left = action[0][1]
            vm.mask_right = action[0][1]
            action.pop(0)
        elif action[0][0] == "blur":
            vm.mask_blur = int(action[0][1])
            action.pop(0)    
        elif action [0][0] == "GFPGAN_blend":
            vm.GFPGAN_blend = action[0][1]
            action.pop(0)
        elif action [0][0] == "fake_diff_blend":
            vm.fake_diff_blend = action[0][1]
            action.pop(0)           
        elif action [0][0] == "num_threads":
            vm.num_threads = action[0][1]
            action.pop(0)         
        elif action [0][0] == "face_thresh":
            vm.face_thresh = action[0][1]
            action.pop(0)   
        elif action [0][0] == "apply_CLIPs":
            vm.CLIPs = action[0][1]
            action.pop(0)   
        elif action [0][0] == "CLIP_checkbox":
            vm.toggle_CLIPs = action[0][1]
            action.pop(0)  
        elif action [0][0] == "pos_thresh":
            vm.pos_thresh = action[0][1]
            action.pop(0)              
        elif action [0][0] == "neg_thresh":
            vm.neg_thresh = action[0][1]
            action.pop(0) 
        elif action [0][0] == "saved_video_path":
            vm.saved_video_path = action[0][1]
            action.pop(0) 
        elif action [0][0] == "CLIP_blur":
            vm.CLIP_blur = int(action[0][1])
            action.pop(0)   
        elif action [0][0] == "toggle_occluder":
            vm.occluder = int(action[0][1])
            action.pop(0) 
        elif action [0][0] == "occluder_blur":
            vm.occluder_blur = int(action[0][1])
            action.pop(0) 
        elif action [0][0] == "occluder_limit":
            vm.occluder_limit = int(action[0][1])
            action.pop(0)  
        
          
        elif action [0][0] == "load_models":
            gui.set_status("loading GFPGAN...")
            GFPGAN_session = load_GFPGAN_model()
            vm.set_GFPGAN_model(GFPGAN_session)
            gui.set_status("loading Swapper...")
            swapper, emap = load_swapper_model()
            vm.set_swapper_model(swapper, emap)
            gui.set_status("loading Faceapp...")
            faceapp = load_faceapp_model() 
            gui.set_faceapp_model(faceapp)
            vm.set_faceapp_model(faceapp)  
            gui.set_status("loading txt2CLIP...")
            vm.clip_session, vm.cuda_device = load_clip_model()
            gui.set_status("loading Occuluder...")
            vm.occluder_model, vm.occluder_tensor = load_occluder_model()
            gui.set_status("loading Target Videos...")
            gui.populate_target_videos()
            gui.set_status("loading Source Faces...")
            gui.populate_faces_canvas()
            gui.set_status("Done...")
            action.pop(0)    

            
        # From VM    
        elif action[0][0] == "stop_play":
            gui.toggle_play_video()
            action.pop(0)        
        
        elif action[0][0] == "set_slider_length":
            gui.set_video_slider_length(action[0][1])
            action.pop(0)
  
        elif action[0][0] == "send_msg":    
            gui.set_status(action[0][1])
            action.pop(0) 
            
        else:
            print("Action not found: "+action[0][0]+" "+str(action[0][1]))
            action.pop(0)

      # start = time.time()
  

    gui.check_for_video_resize()
    vm.process()
    gui.after(1, coordinator)
    # print(time.time() - start)    
    
def load_faceapp_model():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def load_swapper_model():    
    # Load Swapper model and get graph param
    model = onnx.load("./inswapper_128.fp16.onnx")
    graph = model.graph

    emap = onnx.numpy_helper.to_array(graph.initializer[-1])
    
    # Create Swapper model session
    opts = onnxruntime.SessionOptions()
    # opts.enable_profiling = True 
    return onnxruntime.InferenceSession( "./inswapper_128.fp16.onnx", opts, providers=["CUDAExecutionProvider"]), emap
    
def load_clip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")    
    clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    clip_session.eval();
    clip_session.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device('cuda')), strict=False) 
    clip_session.to(device)    
    return clip_session, device

def load_GFPGAN_model():
    global GFPGAN_session
    GFPGAN_session = onnxruntime.InferenceSession( "./GFPGANv1.4.onnx", providers=["CUDAExecutionProvider"])
    return GFPGAN_session
    
def load_occluder_model():            
    to_tensor = transforms.ToTensor()
    model = smp.Unet(encoder_name='resnet18', encoder_weights='imagenet', classes=1, activation=None)

    weights = torch.load('./occluder.ckpt')
    new_weights = OrderedDict()
    for key in weights.keys():
        new_key = '.'.join(key.split('.')[1:])
        new_weights[new_key] = weights[key]

    model.load_state_dict(new_weights)
    model.to('cuda')
    model.eval()
    return model, to_tensor
    
def run():
    global gui, vm, faceapp, action, frame, swapper, r_frame, GFPGAN_session, clip_session
    gui = GUI.GUI()
    vm = VM.VideoManager()
    GFPGAN_session = []
    faceapp = [] 
    swapper = []    
    action = []
    frame = []
    r_frame = []
        

    gui.initialize_gui() 
    coordinator()    
    
    gui.mainloop()


