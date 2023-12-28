# #!/usr/bin/env python3
import os
import time
from collections import OrderedDict

import rope.GUI as GUI
import rope.VideoManager as VM

import onnxruntime
import onnx
import torch
from torchvision import transforms

from rope.external.clipseg import CLIPDensePredT

onnxruntime.set_default_logger_severity(4)

resize_delay = 1

# @profile
def coordinator():
    global gui, vm, action, frame, r_frame, load_notice, resize_delay
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
        frame.pop(0)
 ####################   
    if vm.get_requested_frame_length() > 0:
        r_frame.append(vm.get_requested_frame())    
    if len(r_frame) > 0:
        gui.set_image(r_frame[0], True)
        r_frame=[]
 ####################   
    if len(action) > 0:
        # print('Action:', action[0][0])
        if action[0][0] == "load_target_video":
            vm.load_target_video(action[0][1])
            action.pop(0)
        elif action[0][0] == "load_target_image":
            vm.load_target_image(action[0][1])
            action.pop(0)            
        elif action[0][0] == "play_video":
            vm.play_video(action[0][1])
            action.pop(0)
        elif action[0][0] == "get_requested_video_frame":
            # print(gui.video_slider.get())
            vm.get_requested_video_frame(action[0][1])
            action.pop(0)
        elif action[0][0] == "get_requested_video_frame_parameters":
            vm.get_requested_video_frame_parameters(action[0][1])
            action.pop(0)    
        elif action[0][0] == "get_requested_image":
            vm.get_requested_image()
            action.pop(0)            
        elif action[0][0] == "find_faces":
            gui.find_faces(action[0][1])
            action.pop(0)    
        elif action[0][0] == "clear_faces":
            gui.clear_faces()
            action.pop(0)    
        elif action[0][0] == "swap":
            if not vm.swapper_model:
                swapper, emap = load_swapper_model()
                vm.set_swapper_model(swapper, emap)
                if not vm.detection_model:
                    vm.detection_model = load_detection_model()
                    vm.recognition_model = load_recognition_model()

            vm.swap = action[0][1]
            action.pop(0)
        elif action[0][0] == "target_faces":
            vm.target_facess = action[0][1]
            action.pop(0)
        elif action [0][0] == "num_threads":
            vm.num_threads = action[0][1]
            action.pop(0)         
        elif action [0][0] == "saved_video_path":
            vm.saved_video_path = action[0][1]
            action.pop(0) 
        elif action [0][0] == "vid_qual":
            vm.vid_qual = int(action[0][1])
            action.pop(0) 
        elif action [0][0] == "set_stop":
            vm.stop_marker = action[0][1]
            action.pop(0)  
        elif action [0][0] == "perf_test":
            vm.perf_test = action[0][1]
            action.pop(0)               
        # elif action [0][0] == "load_null":
            # vm.load_null()
            # action.pop(0) 
        elif action [0][0] == "parameters":
            if action[0][1]['UpscaleState']:
                if not vm.resnet_model:
                    vm.resnet_model = load_resnet_model()  
                index = action[0][1]['UpscaleMode']
                if action[0][1]['UpscaleModes'][index] == 'GFPGAN':
                    if not vm.GFPGAN_model:
                        vm.GFPGAN_model = load_GFPGAN_model()
                elif action[0][1]['UpscaleModes'][index] == 'CF':
                    if not vm.codeformer_model:
                        vm.codeformer_model = load_codeformer_model()
                elif action[0][1]['UpscaleModes'][index] == 'GPEN256':
                    if not vm.GPEN_256_model:
                        vm.GPEN_256_model = load_GPEN_256_model()                
                elif action[0][1]['UpscaleModes'][index] == 'GPEN512':
                    if not vm.GPEN_512_model:
                        vm.GPEN_512_model = load_GPEN_512_model()
            if action[0][1]["CLIPState"]:
                if not vm.clip_session:
                    vm.clip_session = load_clip_model()
            if action[0][1]["OccluderState"]:
                if not vm.occluder_model:
                    vm.occluder_model = load_occluder_model()
            if action[0][1]["FaceParserState"]:
                if not vm.face_parsing_model:
                    vm.face_parsing_model, vm.face_parsing_tensor = load_face_parser_model()
            vm.parameters = action[0][1]
            action.pop(0) 
        elif action [0][0] == "markers":
            vm.markers = action[0][1]
            action.pop(0)            
        elif action[0][0] == 'load_faceapp_model':
            if not vm.detection_model or not gui.detection_model:
                detection_model = load_detection_model()
                vm.detection_model = detection_model
                gui.detection_model = detection_model

                recognition_model = load_recognition_model()
                vm.recognition_model = recognition_model
                gui.recognition_model = recognition_model

            action.pop(0)
            
        elif action [0][0] == "load_models":
            gui.populate_target_videos()
            gui.load_source_faces()
            action.pop(0)  
        elif action[0][0] == "function":
            eval(action[0][1])
            action.pop(0)
        elif action [0][0] == "clear_mem":
            vm.clear_mem()
            action.pop(0)    

            
        # From VM    
        elif action[0][0] == "stop_play":
            gui.set_player_buttons_to_inactive()
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
  

    if resize_delay > 5:
        gui.check_for_video_resize()
        resize_delay = 0
    else:
        resize_delay +=1
        
    vm.process()
    gui.after(1, coordinator)
    # print(time.time() - start)    
    


def load_swapper_model():    
    model = onnx.load("./models/inswapper_128.fp16.onnx")
    
    graph = model.graph
    emap = onnx.numpy_helper.to_array(graph.initializer[-1])
    
    sess_options = onnxruntime.SessionOptions()
    # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    
    
    return onnxruntime.InferenceSession( "./models/inswapper_128.fp16.onnx", sess_options, providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']), emap
    
def load_clip_model():
    # https://github.com/timojl/clipseg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    # clip_session = CLIPDensePredTMasked(version='ViT-B/16', reduce_dim=64)
    clip_session.eval();
    clip_session.load_state_dict(torch.load('./models/rd64-uni-refined.pth'), strict=False) 
    clip_session.to(device)    
    return clip_session 

def load_GPEN_512_model():
    session = onnxruntime.InferenceSession( "./models/GPEN-BFR-512.onnx", providers=["CUDAExecutionProvider", 'CPUExecutionProvider'])
    return session

def load_GPEN_256_model():
    session = onnxruntime.InferenceSession( "./models/GPEN-BFR-256.onnx", providers=["CUDAExecutionProvider", 'CPUExecutionProvider'])
    return session

def load_GFPGAN_model():
    session = onnxruntime.InferenceSession( "./models/GFPGANv1.4.onnx", providers=["CUDAExecutionProvider", 'CPUExecutionProvider'])
    return session
    
def load_codeformer_model():    
    codeformer_session = onnxruntime.InferenceSession( "./models/codeformer_fp16.onnx", providers=["CUDAExecutionProvider", 'CPUExecutionProvider'])
    return codeformer_session

def load_occluder_model():            
    model = onnxruntime.InferenceSession("./models/occluder.onnx", providers=["CUDAExecutionProvider", 'CPUExecutionProvider'])
    return model 

def load_face_parser_model():    
    session = onnxruntime.InferenceSession("./models/faceparser_fp16.onnx", providers=["CUDAExecutionProvider", 'CPUExecutionProvider'])

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    return session, to_tensor 
    
def load_resnet_model():     
    model = onnxruntime.InferenceSession("./models/res50.onnx", providers=["CUDAExecutionProvider", 'CPUExecutionProvider'])
    return model   
    
def load_detection_model():
    session = onnxruntime.InferenceSession('.\models\det_10g.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return session

def load_recognition_model():
    session = onnxruntime.InferenceSession('.\models\w600k_r50.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return session
    
    
def run():
    global gui, vm, action, frame, r_frame, resize_delay

    
    gui = GUI.GUI()
    vm = VM.VideoManager()

    action = []
    frame = []
    r_frame = []

    gui.initialize_gui() 
    

    coordinator()    
    
    gui.mainloop()


