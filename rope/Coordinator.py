# #!/usr/bin/env python3

import time
import torch
from torchvision import transforms

import rope.GUI as GUI
import rope.VideoManager as VM
import rope.Models as Models
from rope.external.clipseg import CLIPDensePredT

resize_delay = 1
mem_delay = 1

# @profile
def coordinator():
    global gui, vm, action, frame, r_frame, load_notice, resize_delay, mem_delay
    # start = time.time()

  
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
        # print('Value:', action[0][1])
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
            vm.get_requested_video_frame(action[0][1], marker=False)
            action.pop(0)
        elif action[0][0] == "get_requested_video_frame_without_markers":
            vm.get_requested_video_frame(action[0][1], marker=False)
            action.pop(0)    
        elif action[0][0] == "get_requested_image":
            vm.get_requested_image()
            action.pop(0)            
        elif action[0][0] == "swap":

            vm.swap = action[0][1]
            action.pop(0)
        elif action[0][0] == "target_faces":
            vm.assign_found_faces(action[0][1])
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
        elif action [0][0] == 'ui_vars':
            vm.ui_data = action[0][1]
            action.pop(0) 
        elif action [0][0] == 'control':
            vm.control = action[0][1]
            action.pop(0) 
        elif action [0][0] == "parameters":
            if action[0][1]["CLIPSwitch"]:
                if not vm.clip_session:
                    vm.clip_session = load_clip_model()

            vm.parameters = action[0][1]
            action.pop(0) 
        elif action [0][0] == "markers":
            vm.markers = action[0][1]
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

          
            
        else:
            print("Action not found: "+action[0][0]+" "+str(action[0][1]))
            action.pop(0)


  

    if resize_delay > 100:
        gui.check_for_video_resize()
        resize_delay = 0
    else:
        resize_delay +=1
        
    if mem_delay > 1000:
        gui.update_vram_indicator()
        mem_delay = 0
    else:
        mem_delay +=1
        
    vm.process()
    gui.after(1, coordinator)
    # print(time.time() - start)    
    



    
def load_clip_model():
    # https://github.com/timojl/clipseg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    # clip_session = CLIPDensePredTMasked(version='ViT-B/16', reduce_dim=64)
    clip_session.eval();
    clip_session.load_state_dict(torch.load('./models/rd64-uni-refined.pth'), strict=False) 
    clip_session.to(device)    
    return clip_session 


    
    
def run():
    global gui, vm, action, frame, r_frame, resize_delay, mem_delay

    models = Models.Models()
    gui = GUI.GUI(models)
    vm = VM.VideoManager(models)
    

    action = []
    frame = []
    r_frame = []

    gui.initialize_gui() 
    

    coordinator()    
    
    gui.mainloop()


