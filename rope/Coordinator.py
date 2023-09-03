# #!/usr/bin/env python3
import os
import time

import rope.GUI as GUI
import rope.VideoManager as VM

from insightface.app import FaceAnalysis
import onnxruntime
import onnx

import torch
from rope.external.clipseg import CLIPDensePredT

import segmentation_models_pytorch as smp
from collections import OrderedDict
from torchvision import transforms

from rope.external.model import BiSeNet



def coordinator():
    global gui, vm, action, frame, r_frame
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
            if not vm.swapper_model:
                gui.set_status("loading Swapper")
                swapper, emap = load_swapper_model()
                vm.set_swapper_model(swapper, emap)
                gui.set_status("Swapper loaded!")   
            vm.swap_set(action[0][1])
            action.pop(0)
        elif action[0][0] == "source_embeddings":  
            vm.load_source_embeddings(action[0][1])
            action.pop(0)
        elif action[0][0] == "target_faces":
            vm.found_faces_assignments = action[0][1]
            action.pop(0)

   

        
        elif action [0][0] == "num_threads":
            vm.num_threads = action[0][1]
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



        elif action [0][0] == "vid_qual":
            vm.vid_qual = int(action[0][1])
            action.pop(0) 

        elif action [0][0] == "parameters":
            if action[0][1]["GFPGANState"]:
                if not vm.GFPGAN_model:
                    gui.set_status("loading GFPGAN...")
                    vm.GFPGAN_model = load_GFPGAN_model()
                    gui.set_status("GFPGAN loaded!")
            if action[0][1]["CLIPState"]:
                if not vm.clip_session:
                    gui.set_status("loading CLIP..")
                    vm.clip_session, vm.cuda_device = load_clip_model()
                    gui.set_status("CLIP loaded!")                   
            if action[0][1]["OccluderState"]:
                if not vm.occluder_model:
                    gui.set_status("loading Occluder.")
                    vm.occluder_model, vm.occluder_tensor = load_occluder_model()
                    gui.set_status("Occluder loaded!")                     
            if action[0][1]["FaceParserState"]:
                if not vm.face_parsing_model:
                    gui.set_status("loading FaceParser")
                    vm.face_parsing_model, vm.face_parsing_tensor = load_face_parser_model()
                    gui.set_status("FaceParser loaded!")                       
                   
                    
            vm.parameters = action[0][1]
            action.pop(0) 
            
        elif action [0][0] == "load_models":
            gui.set_status("loading Faceapp...")
            faceapp = load_faceapp_model() 
            gui.set_faceapp_model(faceapp)
            vm.set_faceapp_model(faceapp)  
            gui.set_status("loading Target Videos...")
            gui.populate_target_videos()
            gui.set_status("loading Source Faces...")
            gui.load_source_faces()
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
    model = onnx.load("./models/inswapper_128.fp16.onnx")
    graph = model.graph

    emap = onnx.numpy_helper.to_array(graph.initializer[-1])
    
    # Create Swapper model session
    opts = onnxruntime.SessionOptions()
    # opts.enable_profiling = True
    opts.enable_cpu_mem_arena = False
    return onnxruntime.InferenceSession( "./models/inswapper_128.fp16.onnx", opts, providers=["CUDAExecutionProvider"]), emap
    
def load_clip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")    
    clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    clip_session.eval();
    clip_session.load_state_dict(torch.load('./models/rd64-uni-refined.pth', map_location=torch.device('cuda')), strict=False) 
    clip_session.to(device)    
    return clip_session, device

def load_GFPGAN_model():

    opts = onnxruntime.SessionOptions()
    # opts.enable_profiling = True
    opts.enable_cpu_mem_arena = False
    GFPGAN_session = onnxruntime.InferenceSession( "./models/GFPGANv1.4.onnx", providers=["CUDAExecutionProvider"])
    return GFPGAN_session
    
def load_occluder_model():            
    to_tensor = transforms.ToTensor()
    model = smp.Unet(encoder_name='resnet18', encoder_weights='imagenet', classes=1, activation=None)

    weights = torch.load('./models/occluder.ckpt')
    new_weights = OrderedDict()
    for key in weights.keys():
        new_key = '.'.join(key.split('.')[1:])
        new_weights[new_key] = weights[key]

    model.load_state_dict(new_weights)
    model.to('cuda')
    model.eval()
    return model, to_tensor  

def load_face_parser_model():    
    n_classes = 19
    model = BiSeNet(n_classes=n_classes)
    model.cuda()
    model.load_state_dict(torch.load("./models/79999_iter.pth"))
    model.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    return model, to_tensor  
    
def run():
    global gui, vm, action, frame, r_frame
    gui = GUI.GUI()
    vm = VM.VideoManager()

    action = []
    frame = []
    r_frame = []
        

    gui.initialize_gui() 
    coordinator()    
    
    gui.mainloop()


