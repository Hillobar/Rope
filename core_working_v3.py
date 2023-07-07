# #!/usr/bin/env python3
import os
import queue
import time

import roop.GUI as GUI
import roop.VideoManager as VM
import roop.GFPGANFaceAugment as GFA

import insightface
import onnxruntime
import onnx



def coordinator():
    global gui, vm, faceapp, swapper, action, frame, r_frame, faceaugment
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
        gui.set_image(r_frame[0], True)
        gui.display_image_in_video_frame()
        r_frame.pop(0)
 ####################   
    if len(action) > 0:
        # From GUI
        if action[0][0] == "load_source_faces":
            if not faceapp:
                faceapp = load_faceapp_model()
                gui.set_faceapp_model(faceapp)
                vm.set_faceapp_model(faceapp)
            gui.populate_faces_canvas()
            action.pop(0)
        # elif action[0][0] == "load_target_videos": # should this be completely internal to GUI?>
            # gui.populate_target_videos("D:\SD\Videos\Input\Videos")
            # action.pop(0)
        # Select Target Video and place in preview
        elif action[0][0] == "load_target_video":
            vm.load_target_video(action[0][1])
            gui.set_slider_position(0)
            action.pop(0)
        elif action[0][0] == "play_video":
            vm.play_video(action[0][1])
            action.pop(0)
        elif action[0][0] == "set_video_position":
            vm.get_requested_video_frame(action[0][1])
            action.pop(0)
        elif action[0][0] == "find_faces":
            if not faceapp:
                faceapp = load_faceapp_model() 
                gui.set_faceapp_model(faceapp)
                vm.set_faceapp_model(faceapp) 
            gui.find_faces(action[0][1])
            action.pop(0)    
        elif action[0][0] == "clear_faces":
            gui.clear_faces()
            action.pop(0)    
        elif action[0][0] == "swap":
            if not swapper:
                swapper, emap = load_swapper_model()
                vm.set_swapper_model(swapper, emap)
            if not faceapp:
                faceapp = load_faceapp_model() 
                gui.set_faceapp_model(faceapp)
                vm.set_faceapp_model(faceapp)                
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
            if not faceaugment:
                faceaugment = load_GFPGAN_model()
                vm.set_GFPGAN_model(faceaugment)
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
            action.pop(0)
        elif action[0][0] == "right_blend":
            vm.mask_right = action[0][1]
            action.pop(0)  
        elif action[0][0] == "blur":
            vm.mask_blur = action[0][1]
            action.pop(0)    
        elif action [0][0] == "GFPGAN_blend":
            vm.GFPGAN_blend = action[0][1]
            action.pop(0)
        elif action [0][0] == "fake_diff_blend":
            vm.fake_diff_blend = action[0][1]
            action.pop(0)           
            
            
            
            
        # From VM    
        elif action[0][0] == "set_slider_length":
            gui.set_video_slider_length(action[0][1])
            action.pop(0)
        else:
            print("Action not found: "+action[0][0]+" "+str(action[0][1]))
            action.pop(0)

      # start = time.time()
  

    gui.check_for_video_resize()
###########################    
    vm.process()
###########################    
    # gui.process()
    gui.after(1, coordinator)
    # print(time.time() - start)     
def load_faceapp_model():
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def load_swapper_model():    
    # Load Swapper model and get graph param
    model = onnx.load("./inswapper_128.onnx")
    graph = model.graph

    emap = onnx.numpy_helper.to_array(graph.initializer[-1])
    
    # Create Swapper model session
    opts = onnxruntime.SessionOptions()
    # opts.enable_profiling = True 
    return onnxruntime.InferenceSession( "./inswapper_128.onnx", opts, providers=["CUDAExecutionProvider"]), emap

def load_GFPGAN_model():
    global faceaugment
    faceaugment = GFA.GFPGANFaceAugment()
    return faceaugment
    
def run():
    global gui, vm, faceapp, action, frame, swapper, r_frame, faceaugment
    gui = GUI.GUI()
    vm = VM.VideoManager()
    faceaugment = []
    faceapp = [] 
    swapper = []    
    action = []
    frame = []
    r_frame = []

    gui.initialize_gui() 
    coordinator()    
    
    gui.mainloop()


