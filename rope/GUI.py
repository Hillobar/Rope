import os
import cv2
import tkinter as tk
from tkinter import filedialog, font
import numpy as np
from PIL import Image, ImageTk
import json
import time
from skimage import transform as trans
from math import floor, ceil
import copy
import bisect
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import mimetypes
import webbrowser
#import inspect print(inspect.currentframe().f_back.f_code.co_name, 'resize_image')

from  rope.Dicts import PARAM_BUTTONS_PARAMS, ACTIONS, PARAM_BUTTONS_CONSTANT
last_frame = 0

class GUI(tk.Tk):
    def __init__( self):  
        super().__init__()
        # Adding a title to the self
        # self.call('tk', 'scaling', 0.5)
        self.title('Rope-Ruby-03')
        self.pixel = []
        self.parameters = PARAM_BUTTONS_PARAMS
        self.actions = ACTIONS
        self.param_const = PARAM_BUTTONS_CONSTANT
        self.parameters_buttons={}
        self.num_threads = 1
        self.video_quality = 18
        self.target_media = []
        self.target_video_file = []
        self.action_q = []
        self.video_image = []
        self.x1 = []
        self.y1 = []
        self.found_faces_assignments = []
        self.play_video = False
        self.rec_video = False
        # self.faceapp_model = []
        self.video_loaded = False
        self.docked = True
        self.undock = []
        self.image_file_name = []
        self.stop_marker = []
        self.stop_image = []
        self.marker_icon = []
        self.stop_marker_icon = []
        self.video_length = []

        # self.window_y = []
        # self.window_width = []
        # self.window_height = []
        self.detection_model = []
        self.recognition_model = []

        self.window_last_change = []


        
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)   

        self.json_dict =    {
                            "source videos":    None, 
                            "source faces":     None, 
                            "saved videos":     None, 
                            "threads":          1, 
                            'dock_win_geom':    [980, 1020, self.winfo_screenwidth()/2-400, self.winfo_screenheight()/2-510],
                            'undock_win_geom':  [980, 517, self.winfo_screenwidth()/2-400, self.winfo_screenheight()/2-510],
                            'player_geom':      [1024, 768, self.winfo_screenwidth()/2-400, self.winfo_screenheight()/2-510],
                            }

        self.marker =  {
                        'frame':        '0',
                        'parameters':   '',
                        'icon_ref':     '',
                        }
        self.markers = []   

        self.target_face = {    
                            "TKButton":                 [],
                            "ButtonState":              "off",
                            "Image":                    [],
                            "Embedding":                [],
                            "SourceFaceAssignments":    [],
                            "EmbeddingNumber":          0,       #used for adding additional found faces
                            'AssignedEmbedding':        [],     #the currently assigned source embedding, including averaged ones
                            }
        self.target_faces = []
        
        self.source_face =  {
                            "TKButton":                 [],
                            "ButtonState":              "off",
                            "Image":                    [],
                            "Embedding":                []
                            }   
        self.source_faces = []        

        self.button_highlight_style =    {  
                                'bg':               'light goldenrod', 
                                'fg':               'black', 
                                'activebackground': 'gray75', 
                                'activeforeground': 'light goldenrod',
                                'relief':           'flat',
                                'border':           '0',
                                'font':             ("Cascadia Mono Light", 9)
                                }   
        self.inactive_button_style =    {  
                                'bg':               'gray20', 
                                'fg':               'white', 
                                'activebackground': 'gray10', 
                                'activeforeground': 'white',
                                'relief':           'flat',
                                'border':           '0',
                                'font':             ("Cascadia Mono Light", 9)
                                }   
        self.active_button_style =    {  
                                'bg':               'black', 
                                'fg':               'white', 
                                'activebackground': 'gray10', 
                                'activeforeground': 'white',
                                'relief':           'flat',
                                'border':           '0',
                                'font':             ("Cascadia Mono Light", 9)
                                }
        self.need_button_style =    {  
                                'bg':               'gray30', 
                                'fg':               'white', 
                                'relief':           'flat',
                                'border':           '0',                                	
                                'font':             ("Cascadia Mono Light", 9)
                                } 

        self.canvas_label_style =    {  
                                'bg':               'gray20', 
                                'relief':           'flat',
                                'bd':               '0',
                                'highlightthickness': '0'
                                }                                 
                                
        self.canvas_style1 =    {  
                                'bg':               'gray20', 
                                'relief':           'flat',
                                'bd':               '0',
                                'highlightthickness': '0'
                                }                 
        self.frame_style =      {  
                                'bg':               'gray20', 
                                'relief':           'flat',
                                'bd':               '0'
                                }  
        self.checkbox_style =   {  
                                'bg':               'gray40', 
                                'fg':               'white',
                                'relief':           'flat',
                                'bd':               '0',
                                'anchor':           'w',
                                'selectcolor':      'gray40' 
                                }   
        self.label_style =      {  
                                'bg':               'gray20', 
                                'fg':               'white',                                
                                'relief':           'flat',
                                'bd':               '0',
                                'font':             ("Cascadia Mono Light", 9),
                                'anchor':           'w'                                
                                }      
        self.slider_style =     {  
                                'bg':               'gray20', 
                                'fg':               'white', 
                                'activebackground': 'black', 
                                'highlightthickness':'0',
                                'relief':           'flat',
                                'sliderrelief':     'flat',                                
                                'border':           '0',
                                'width':            '10',
                                'troughcolor':      'gray40',
                                'font':             ("Cascadia Mono Light", 9)
                                }                                    
                                                    
        # Pop up window
        self.undock_video_window = tk.Toplevel()
        self.undock_video_window.withdraw()           
        
        # Undocked Media frame
        self.undock_video_frame = tk.Frame( self.undock_video_window, self.frame_style)
        self.undock_video_frame.grid( row = 0, column = 0, sticky='NEWS', pady = 2 )
        
        self.undock_video_frame.grid_columnconfigure(0, weight=1)  
        self.undock_video_frame.grid_rowconfigure(0, weight = 1)
        
        # Video [0,0]
        self.undock_video = tk.Label( self.undock_video_frame, self.label_style, bg='black')
        self.undock_video.grid( row = 0, column = 0, sticky='NEWS', pady =0 )
        self.undock_video.bind("<MouseWheel>", self.iterate_through_merged_embeddings)
        self.undock_video.bind("<ButtonRelease-1>", lambda event: self.toggle_play_video())
        

        
        # Docked Media frame
        self.video_frame = tk.Frame( self, self.frame_style)
        self.video_frame.grid( row = 0, column = 0, sticky='NEWS', pady = 0 )
        
        self.video_frame.grid_columnconfigure(0, weight=1)  
        self.video_frame.grid_rowconfigure(0, weight = 1)
    
        # Video [0,0]
        self.video = tk.Label( self.video_frame, self.label_style, bg='black')
        self.video.grid( row = 0, column = 0, sticky='NEWS', pady =0 )
        self.video.bind("<MouseWheel>", self.iterate_through_merged_embeddings)
        self.video.bind("<ButtonRelease-1>", lambda event: self.toggle_play_video())
        
 ######### Options 
        # Play bar
        self.options_frame = tk.Frame( self, self.frame_style, height = 71)
        self.options_frame.grid( row = 1, column = 0, sticky='NEWS', pady = 2 )
        
        self.options_frame.grid_rowconfigure( 0, weight = 100 )  
        self.options_frame.grid_rowconfigure( 1, weight = 100 )         
        self.options_frame.grid_columnconfigure( 0, weight = 1 ) 
        
        # Media control canvas
        self.media_control_canvas = tk.Canvas( self.options_frame, self.canvas_style1, height = 40)
        self.media_control_canvas.grid( row = 0, column = 0, sticky='NEWS', pady = 0)  
        self.media_control_canvas.grid_columnconfigure(1, weight = 1) 

        # Video button canvas
        self.video_button_canvas = tk.Canvas( self.media_control_canvas, self.canvas_style1, width = 112, height = 40)
        self.video_button_canvas.grid( row = 0, column = 0, sticky='NEWS', pady = 0)       

        # Buttons
        self.create_ui_button_2('Dock', self.video_button_canvas, 8, 2, width=15, height=36)
        self.create_ui_button_2('Play', self.video_button_canvas, 31, 2, width=36, height=36)
        self.create_ui_button_2('Record', self.video_button_canvas, 69, 2, width=36, height=36)

        # Video Slider canvas
        self.video_slider_canvas = tk.Canvas( self.media_control_canvas, self.canvas_style1, height=40)
        self.video_slider_canvas.grid( row = 0, column = 1, sticky='NEWS', pady = 0)  

        # Video Slider
        self.video_slider = tk.Scale( self.video_slider_canvas, self.slider_style, orient='horizontal')
        self.video_slider.bind("<B1-Motion>", lambda event: self.slider_move('motion', self.video_slider.get()))
        self.video_slider.bind("<ButtonPress-1>", lambda event: self.slider_move('press', self.video_slider.get()))
        self.video_slider.bind("<ButtonRelease-1>", lambda event: self.slider_move('release', self.video_slider.get()))
        self.video_slider.bind("<ButtonRelease-3>", lambda event: self.slider_move('motion', self.video_slider.get()))
        self.video_slider.bind("<MouseWheel>", lambda event: self.mouse_wheel(event, self.video_slider.get()))
        self.video_slider.pack(fill=tk.X)

        # Marker canvas
        self.marker_canvas = tk.Canvas( self.media_control_canvas, self.canvas_style1, width = 180, height = 40)
        self.marker_canvas.grid( row = 0, column = 2, sticky='news', pady = 0)        
        
        # Marker Buttons
        self.create_ui_button_2('AddMarker', self.marker_canvas, 8, 2, width=36, height=36)
        self.create_ui_button_2('RemoveMarker', self.marker_canvas, 35, 2, width=36, height=36)
        self.create_ui_button_2('PrevMarker', self.marker_canvas, 69, 2, width=36, height=36)
        self.create_ui_button_2('NextMarker', self.marker_canvas, 107, 2, width=36, height=36)
        self.create_ui_button_2('ToggleStop', self.marker_canvas, 140, 2, width=36, height=36)

        # Image control canvas
        self.image_control_canvas = tk.Canvas( self.video_frame, self.canvas_style1, height = 40)
        self.image_control_canvas.grid( row = 1, column = 0, sticky='NEWS', pady = 0)  
        self.image_control_canvas.grid_columnconfigure(1, weight = 1)        
        
        # Image Save
        self.create_ui_button_2('ImgDock', self.image_control_canvas, 8, 2, width=15, height=36)
        self.create_ui_button_2('SaveImage', self.image_control_canvas, 31, 2, width=36, height=36)
        self.create_ui_button_2('AutoSwap', self.image_control_canvas, 65, 2, width=36, height=36)

        
        # Options Area
        x_space = 40
        # Left Canvas
        self.options_frame_canvas1 = tk.Canvas( self.options_frame, self.canvas_style1, height = 71)
        self.options_frame_canvas1.grid( row = 1, column = 0, sticky='NEWS', pady = 0 )

        # Label Frame 1
        self.label_frame1 = tk.LabelFrame( self.options_frame_canvas1, self.frame_style, height = 71, width = 1400 )
        self.label_frame1.place(x=0, y=0)
        
        column=8
        self.create_ui_button('Upscale', self.label_frame1, column, 8)
        self.create_ui_button('Threshold', self.label_frame1, column, 37) 

        column=column+125+x_space
        self.create_ui_button('Strength', self.label_frame1, column, 8)
        self.create_ui_button('Orientation', self.label_frame1, column, 37)        
        
        column=column+125+x_space
        self.create_ui_button('Border', self.label_frame1, column, 8)
        self.create_ui_button('Diff', self.label_frame1, column, 37)
        
        column=column+125+x_space  
        self.create_ui_button('Occluder', self.label_frame1, column, 8)
        self.create_ui_button('FaceParser', self.label_frame1, column, 37)        

        column=column+125+x_space  
        self.create_ui_button('CLIP', self.label_frame1, column, 8)
        # CLIP-entry
        self.temptkstr = tk.StringVar(value="")
        self.CLIP_text = tk.Entry(self.label_frame1, relief='flat', bd=0, textvariable=self.temptkstr)
        self.CLIP_text.place(x=column, y=40, width = 125, height=20) 
        self.CLIP_text.bind("<Return>", lambda event: self.update_CLIP_text(self.temptkstr))
        self.CLIP_name = self.nametowidget(self.CLIP_text)

        column=column+125+x_space
        self.create_ui_button('Blur', self.label_frame1, column, 8)   
        self.create_ui_button('MaskView', self.label_frame1, column, 37)
        
        column=column+125+x_space
        self.create_ui_button('Color', self.label_frame1, column, 8)
        self.create_ui_button('Transform', self.label_frame1, column, 37)   

        column=column+125+x_space
        self.create_ui_button('RefDel', self.label_frame1, column, 8)
         

 ######## Target Faces           
        # Frame
        self.found_faces_frame = tk.Frame( self, self.frame_style)
        self.found_faces_frame.grid( row = 2, column = 0, sticky='NEWS', pady = 2 )        
        self.found_faces_frame.grid_columnconfigure( 0, minsize = 10 ) 
        self.found_faces_frame.grid_columnconfigure( 1, weight = 1 )         
        self.found_faces_frame.grid_rowconfigure( 0, weight = 0 )  
        
        # Canvas
        self.found_faces_buttons_canvas = tk.Canvas( self.found_faces_frame, self.canvas_style1, height = 100, width = 112)
        self.found_faces_buttons_canvas.grid( row = 0, column = 0, )

        # Buttons
        self.create_ui_button_2('FindFaces', self.found_faces_buttons_canvas, 8, 8)
        self.create_ui_button_2('ClearFaces', self.found_faces_buttons_canvas, 8, 37)
        self.create_ui_button_2('SwapFaces', self.found_faces_buttons_canvas, 8, 66)

        # Scroll Canvas
        self.found_faces_canvas = tk.Canvas( self.found_faces_frame, self.canvas_style1, height = 100 )
        self.found_faces_canvas.grid( row = 0, column = 1, sticky='NEWS')
        self.found_faces_canvas.bind("<MouseWheel>", self.target_faces_mouse_wheel)
        self.found_faces_canvas.create_text(8, 45, anchor='w', fill='grey25', font=("Arial italic", 50), text=" Target Faces")
        
   
                
 ######## Source Faces       
        # Frame
        self.source_faces_frame = tk.Frame( self, self.frame_style)
        self.source_faces_frame.grid( row = 3, column = 0, sticky='NEWS', pady = 2 )
        self.source_faces_frame.grid_columnconfigure( 0, minsize = 10 ) 
        self.source_faces_frame.grid_columnconfigure( 1, weight = 1 )         
        self.source_faces_frame.grid_rowconfigure( 0, weight = 0 )  
        
        # Canvas
        self.source_faces_buttons = []
        self.source_button_canvas = tk.Canvas( self.source_faces_frame, self.canvas_style1, height = 100, width = 112)
        self.source_button_canvas.grid( row = 0, column = 0, sticky='NEWS')
        
        # Buttons
        self.create_ui_button_2('LoadSFaces', self.source_button_canvas, 8, 8)
        self.create_ui_button_2('DelEmbed', self.source_button_canvas, 8, 66)
               
        # Merged Embeddings Text
        self.merged_embedding_name = tk.StringVar()
        self.merged_embeddings_text = tk.Entry(self.source_button_canvas, relief='flat', bd=0, textvariable=self.merged_embedding_name)
        self.merged_embeddings_text.place(x=8, y=37, width = 96, height=20) 
        self.merged_embeddings_text.bind("<Return>", lambda event: self.save_selected_source_faces(self.merged_embedding_name)) 
        self.me_name = self.nametowidget(self.merged_embeddings_text)

        # Scroll Canvas
        self.source_faces_canvas = tk.Canvas( self.source_faces_frame, self.canvas_style1, height = 100)
        self.source_faces_canvas.grid( row = 0, column = 1, sticky='NEWS')
        self.source_faces_canvas.bind("<MouseWheel>", self.source_faces_mouse_wheel)
        self.source_faces_canvas.create_text(8, 45, anchor='w', fill='grey25', font=("Arial italic", 50), text=' Source Faces')


######### Target Videos
        # Frame
        self.target_videos_frame = tk.Frame( self, self.frame_style)
        self.target_videos_frame.grid( row = 4, column = 0, sticky='NEWS', pady = 2 )
        
        self.target_videos_frame.grid_columnconfigure( 0, minsize = 10 ) 
        self.target_videos_frame.grid_columnconfigure( 1, weight = 1 )         
        self.target_videos_frame.grid_rowconfigure( 0, weight = 0 )  
        
        # Canvas
        self.target_media_buttons = []
        self.target_button_canvas = tk.Canvas(self.target_videos_frame, self.canvas_style1, height = 100, width = 112)
        self.target_button_canvas.grid( row = 0, column = 0, sticky='NEWS')
        
        # Buttons
        self.create_ui_button_2('LoadTVideos', self.target_button_canvas, 8, 8)        
        self.create_ui_button_2('ImgVid', self.target_button_canvas, 8, 37)        
        # self.create_ui_button_2('HoldFace', self.target_button_canvas, 8, 66)
        
        # Video Canvas [0,1]
        self.target_media_canvas = tk.Canvas( self.target_videos_frame, self.canvas_style1, height = 100)
        self.target_media_canvas.grid( row = 0, column = 1, sticky='NEWS')
        self.target_media_canvas.bind("<MouseWheel>", self.target_videos_mouse_wheel)
        self.target_media_canvas.create_text(8, 45, anchor='w', fill='grey25', font=("Arial italic", 50), text=' Target Videos')

        
        
 ######### Options
        
        self.program_options_frame = tk.Frame( self, self.frame_style, height = 42)
        self.program_options_frame.grid( row = 5, column = 0, sticky='NEWS', pady = 2 )
        
        self.program_options_frame.grid_rowconfigure( 0, weight = 100 )          
        self.program_options_frame.grid_columnconfigure( 0, weight = 1 ) 
        
        # Left Canvas
        self.program_options_frame_canvas = tk.Canvas( self.program_options_frame, self.canvas_style1, height = 42)
        self.program_options_frame_canvas.grid( row = 0, column = 0, sticky='NEWS', pady = 0 )

        # Label Frame 1
        self.program_options_label = tk.LabelFrame( self.program_options_frame_canvas, self.frame_style, height = 42, width = 800 )
        self.program_options_label.place(x=0, y=0)        
        
        # Buttons
        column = 8
        x_space = 40
        self.create_ui_button_2('StartRope', self.program_options_label, column, 8, width = 125, height = 26)
        column=column+125+x_space
        self.create_ui_button_2('OutputFolder', self.program_options_label, column, 8, width = 125, height = 26)
        column=column+125+x_space
        self.create_ui_button_2('Threads', self.program_options_label, column, 8, width = 125, height = 26)
        # column=column+125+x_space
        # self.create_ui_button_2('VideoQuality', self.program_options_label, column, 8,width = 125, height = 26)
        column=column+125+x_space
        self.create_ui_button_2('PerfTest', self.program_options_label, column, 8,width = 125, height = 26)
        column=column+125+x_space
        self.create_ui_button_2('Clearmem', self.program_options_label, column, 8,width = 125, height = 26)
        
        # Status
        self.status_frame = tk.Frame( self, bg='grey40', height = 15)
        self.status_frame.grid( row = 6, column = 0, sticky='NEWS', pady = 2 )
        
        self.status_frame.grid_rowconfigure( 0, weight = 1 )          
        self.status_frame.grid_columnconfigure( 0, weight=1 ) 
        self.status_frame.grid_columnconfigure( 1, weight = 1 )
        self.status_frame.grid_columnconfigure( 2, weight=1 )
        
        self.status_left_label = tk.Label(self.status_frame, fg="white", bg='grey20')
        self.status_left_label.grid( row = 0, column = 0, sticky='NEWS')
        
        self.status_label = tk.Label(self.status_frame, fg="white", bg='grey20')
        self.status_label.grid( row = 0, column = 1, sticky='NEWS')
        
        self.donate_label = tk.Label(self.status_frame, fg="light goldenrod", bg='grey20', text="Donate! (Paypal)  ", anchor='e')
        self.donate_label.grid( row = 0, column = 2, sticky='NEWS')
        self.donate_label.bind("<Button-1>", lambda e: self.callback("https://www.paypal.com/donate/?hosted_button_id=Y5SB9LSXFGRF2"))
    
    def callback(self, url):
        webbrowser.open_new_tab(url)
 
    def target_faces_mouse_wheel(self, event):
        self.found_faces_canvas.xview_scroll(1*int(event.delta/120.0), "units") 
   

    def source_faces_mouse_wheel(self, event):
        self.source_faces_canvas.xview_scroll(1*int(event.delta/120.0), "units")

   
    def target_videos_mouse_wheel(self, event):
        self.target_media_canvas.xview_scroll(1*int(event.delta/120.0), "units")
    # focus_get()
    def key_event(self, event):
        # print(event.char, event.keysym, event.keycode)

        if self.focus_get() != self.CLIP_name and self.focus_get() != self.me_name and self.actions['ImgVidMode'] == 0:
            frame = self.video_slider.get()
            if event.char == ' ':
                self.toggle_play_video()
            elif event.char == 'w':
                frame += 1
                if frame > self.video_length:
                    frame = self.video_length
                self.video_slider.set(frame)
                self.add_action("get_requested_video_frame", frame)
                self.parameter_update_from_marker(frame)
            elif event.char == 's':
                frame -= 1 
                if frame < 0:
                    frame = 0   
                self.video_slider.set(frame)
                self.add_action("get_requested_video_frame", frame)
                self.parameter_update_from_marker(frame)
            elif event.char == 'd':
                frame += 30 
                if frame > self.video_length:
                    frame = self.video_length  
                self.video_slider.set(frame)
                self.add_action("get_requested_video_frame", frame)
                self.parameter_update_from_marker(frame)
            elif event.char == 'a':
                frame -= 30 
                if frame < 0:
                    frame = 0                  
                self.video_slider.set(frame)
                self.add_action("get_requested_video_frame", frame)
                self.parameter_update_from_marker(frame)


    def initialize_gui( self ):
        # check if data.json exists, if not then create it 
        try:
            save_file = open("data.json", "r")
        except:
            with open("data.json", "w") as outfile:
                json.dump(self.json_dict, outfile)
        else:   
            save_file.close()
            
       
        json_object = []
        with open('data.json', 'r') as openfile:
            json_object = json.load(openfile)

        try:
            self.json_dict["source videos"] = json_object["source videos"]
        except KeyError:
            self.actions['LoadTVideosButton'].configure(self.button_highlight_style, text=' Setup')
        else:
            if self.json_dict["source videos"] == None:
                self.actions['LoadTVideosButton'].configure(self.button_highlight_style, text=' Setup')
            else:
                temp = self.json_dict["source videos"]
                temp_len = len(temp)
                temp = ' '+temp[temp_len-9:]             
                self.actions['LoadTVideosButton'].configure(self.inactive_button_style, text=temp) 

        try:
            self.json_dict["source faces"] = json_object["source faces"]
        except KeyError:
            self.actions['LoadSFacesButton'].configure(self.button_highlight_style, text=' Setup')
        else:
            if self.json_dict["source faces"] == None:
                self.actions['LoadSFacesButton'].configure(self.button_highlight_style, text=' Setup')
            else:
                temp = self.json_dict["source faces"]
                temp_len = len(temp)
                temp = ' '+temp[temp_len-9:]            
                self.actions['LoadSFacesButton'].configure(self.inactive_button_style, text=temp)
        
        try:
            self.json_dict["saved videos"] = json_object["saved videos"]
        except KeyError:
            self.actions['OutputFolderButton'].configure(self.button_highlight_style, text=' Setup')
        else:
            if self.json_dict["saved videos"] == None:
                self.actions['OutputFolderButton'].configure(self.button_highlight_style, text=' Setup')
            else:
                temp = self.json_dict["saved videos"]
                temp_len = len(temp)
                temp = ' '+temp[temp_len-9:]             
                self.actions['OutputFolderButton'].configure(self.inactive_button_style, text=temp)
                self.add_action("saved_video_path",self.json_dict["saved videos"])

        try:
            self.json_dict["threads"] = json_object["threads"]
        except KeyError:
            self.change_threads_amount(event)
        else:
            temp = self.json_dict["threads"]
            self.num_threads = int(temp)
            temp = ' Threads           ' + str(self.num_threads)
            self.actions['ThreadsButton'].configure(text=temp)        
            self.add_action("num_threads",int(self.num_threads))
            
        try:
            self.json_dict['dock_win_geom'] = json_object['dock_win_geom']
        except:
            self.json_dict['dock_win_geom'] = self.json_dict['dock_win_geom']
        
        try:
            self.json_dict["undock_win_geom"] = json_object["undock_win_geom"]
        except:
            self.json_dict["undock_win_geom"] = self.json_dict["undock_win_geom"]      

        try:
            self.json_dict["player_geom"] = json_object["player_geom"]
        except:
            self.json_dict["player_geom"] = self.json_dict["player_geom"]              

        self.bind('<Key>', lambda event: self.key_event(event))
        self.bind('<space>', lambda event: self.key_event(event))
        
        self.undock_video_window.bind('<Key>', lambda event: self.key_event(event))
        self.undock_video_window.bind('<space>', lambda event: self.key_event(event))
        
        # self.overrideredirect(True)
        self.configure(bg='grey10')
        self.resizable(width=True, height=True) 
        
        # Initialize the window sizes and positions
        self.geometry('%dx%d+%d+%d' % (self.json_dict['dock_win_geom'][0], self.json_dict['dock_win_geom'][1] , self.json_dict['dock_win_geom'][2], self.json_dict['dock_win_geom'][3]))
        self.window_last_change = self.winfo_geometry()
        # Since the undock window hasnt bee created yet, need to set this directly from json
        self.undock_video_window.geometry('%dx%d+%d+%d' % (self.json_dict['player_geom'][0], self.json_dict['player_geom'][1] , self.json_dict['player_geom'][2], self.json_dict['player_geom'][3])) 
        self.undock_video_last_change = self.undock_video_window.winfo_geometry()
        

         
        
     
        
        self.undock_video_window.grid_columnconfigure(0, weight = 1)  
        self.undock_video_window.grid_rowconfigure(0, weight = 10)

        self.grid_columnconfigure(0, weight = 1)  
        self.grid_rowconfigure(0, weight = 10)
        self.grid_rowconfigure(1, weight = 0)  
        self.grid_rowconfigure(2, weight = 0)  
        self.grid_rowconfigure(3, weight = 0)
        self.grid_rowconfigure(4, weight = 0)    
        self.grid_rowconfigure(5, weight = 0)  
        self.grid_rowconfigure(6, weight = 0) 

        self.image_control_canvas.grid_remove()
        self.media_control_canvas.grid()
        
        self.actions['StartRopeButton'].configure(self.button_highlight_style, text=' Load Rope')
        
        img = Image.open('./rope/media/marker.png')
        resized_image= img.resize((15,30), Image.ANTIALIAS)
        self.marker_icon = ImageTk.PhotoImage(resized_image) 
        
        img = Image.open('./rope/media/stop_marker.png')
        resized_image= img.resize((15,30), Image.ANTIALIAS)
        self.stop_marker_icon = ImageTk.PhotoImage(resized_image) 
        
        class empty:
            def __init__(self):
                self.delta = 0
        event = empty()    

        self.update_ui_button('Upscale')
        self.update_ui_button('Diff')
        self.update_ui_button('Border')
        self.update_ui_button('MaskView')
        self.update_ui_button('CLIP')
        self.update_ui_button('Occluder')
        self.update_ui_button('FaceParser')
        self.update_ui_button('Blur')
        self.update_ui_button('Threshold')
        self.update_ui_button('Strength')
        self.update_ui_button('Orientation')
        self.update_ui_button('RefDel')
        self.update_ui_button('Transform')
        self.update_ui_button('Color')


        # self.change_video_quality(event)
        self.change_threads_amount(event)    
        
        self.add_action("parameters", self.parameters)   
        self.set_status('Welcome to Rope-Ruby!')
            
    def load_all(self):
        if not self.json_dict["source videos"] or not self.json_dict["source faces"]:
            print("Please set faces and videos folders first!")
            return

        self.add_action("load_faceapp_model")
        self.add_action("load_models")
        self.actions['StartRopeButton'].configure(self.inactive_button_style, text=" Rope Loaded")
        
        
    def select_video_path(self):
         
        temp = self.json_dict["source videos"]
         
        self.json_dict["source videos"] = filedialog.askdirectory(title="Select Target Videos Folder", initialdir=temp)
        
        temp = self.json_dict["source videos"]
        temp_len = len(temp)
        temp = ' '+temp[temp_len-9:]
         
        self.actions['LoadTVideosButton'].configure(self.inactive_button_style, text=temp) 
        
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)
            
        self.populate_target_videos()
            
    def select_save_video_path(self):
        temp = self.json_dict["saved videos"]
        
        self.json_dict["saved videos"] = filedialog.askdirectory(title="Select Save Video Folder", initialdir=temp)
        
        temp = self.json_dict["saved videos"]
        temp_len = len(temp)
        temp = ' '+temp[temp_len-9:]
         
        self.actions['OutputFolderButton'].configure(self.inactive_button_style, text=temp) 
        
        self.add_action("saved_video_path",self.json_dict["saved videos"])
        
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)    

    def select_faces_path(self):
        temp = self.json_dict["source faces"]
        
        self.json_dict["source faces"] = filedialog.askdirectory(title="Select Source Faces Folder", initialdir=temp)
        
        temp = self.json_dict["source faces"]
        temp_len = len(temp)
        temp = ' '+temp[temp_len-9:]
        
        self.actions['LoadSFacesButton'].configure(self.inactive_button_style, text=temp)
         
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)
        
        self.load_source_faces()
            
    def load_source_faces(self):
        if not self.detection_model:
            self.add_action('load_faceapp_model')
        
        else:
            self.source_faces = []
            self.source_faces_canvas.delete("all")

            # First load merged embeddings
            try:
                temp0 = []
                with open("merged_embeddings.txt", "r") as embedfile:
                    temp = embedfile.read().splitlines() 

                    for i in range(0, len(temp), 513):
                        to = [temp[i][6:], np.array(temp[i+1:i+513], dtype='float32')]
                        temp0.append(to)

                self.pixel = tk.PhotoImage(height=0, width=0)
                
                for j in range(len(temp0)):
                    new_source_face = self.source_face.copy()
                    self.source_faces.append(new_source_face)
                    
                    self.source_faces[j]["ButtonState"] = False
                    self.source_faces[j]["Embedding"] = temp0[j][1] 
                    self.source_faces[j]["TKButton"] = tk.Button(self.source_faces_canvas, self.inactive_button_style, image=self.pixel, text=temp0[j][0], height=14, width=84, compound='left')

                    self.source_faces[j]["TKButton"].bind("<ButtonRelease-1>", lambda event, arg=j: self.toggle_source_faces_buttons_state(event, arg))
                    self.source_faces[j]["TKButton"].bind("<Shift-ButtonRelease-1>", lambda event, arg=j: self.toggle_source_faces_buttons_state_shift(event, arg))
                    self.source_faces[j]["TKButton"].bind("<MouseWheel>", self.source_faces_mouse_wheel)
                    
                    self.source_faces_canvas.create_window((j//4)*92,8+(22*(j%4)), window = self.source_faces[j]["TKButton"],anchor='nw')            
                    # print((j//4)*92,8+(22*(j%4)))
            except:
                pass

            directory = self.json_dict["source faces"]
            filenames = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(directory) for f in filenames]         
            faces = []
            for file in filenames: # Does not include full path
                # Find all faces and ad to faces[]
                # Guess File type based on extension
                try:
                    file_type = mimetypes.guess_type(file)[0][:5]
                except:
                    print('Unrecognized file type:', file)
                else:
                    # Its an image
                    if file_type == 'image':                
                        img = cv2.imread(file)
                        
                        if img is not None:     
                            img = torch.from_numpy(img).to('cuda')
                            img = img.permute(2,0,1)
                            kpss = self.detect(img, input_size = (640, 640), max_num=1, metric='default')
                            ret = []
                            for i in range(kpss.shape[0]):
                                if kpss is not None:
                                    face_kps = kpss[i]

                                face_emb, img_out = self.recognize(img, face_kps)
                                ret.append([face_kps, face_emb, img_out])

                            if ret:
                                crop = cv2.cvtColor(ret[0][2].cpu().numpy(), cv2.COLOR_BGR2RGB)            
                                crop = cv2.resize( crop, (82, 82))
                                faces.append([crop, ret[0][1]])
                        else:
                            print('Bad file', file) 
            shift_i_len = len(self.source_faces)
            
            # Add faces[] images to buttons
            for i in range(len(faces)):   
                new_source_face = self.source_face.copy()
                self.source_faces.append(new_source_face)
                
                shift_i = i+ shift_i_len
            
                self.source_faces[shift_i]["Image"] = ImageTk.PhotoImage(image=Image.fromarray(faces[i][0]))
                self.source_faces[shift_i]["Embedding"] = faces[i][1]
                self.source_faces[shift_i]["TKButton"] = tk.Button(self.source_faces_canvas, self.inactive_button_style, image= self.source_faces[shift_i]["Image"], height = 86, width = 86)
                self.source_faces[shift_i]["ButtonState"] = False
                
                self.source_faces[shift_i]["TKButton"].bind("<ButtonRelease-1>", lambda event, arg=shift_i: self.toggle_source_faces_buttons_state(event, arg))
                self.source_faces[shift_i]["TKButton"].bind("<Shift-ButtonRelease-1>", lambda event, arg=shift_i: self.toggle_source_faces_buttons_state_shift(event, arg))
                self.source_faces[shift_i]["TKButton"].bind("<MouseWheel>", self.source_faces_mouse_wheel)
                
                self.source_faces_canvas.create_window(((shift_i_len//4)+i+1)*92,8, window = self.source_faces[shift_i]["TKButton"],anchor='nw')

            self.source_faces_canvas.configure(scrollregion = self.source_faces_canvas.bbox("all"))
            self.source_faces_canvas.xview_moveto(0)
        
    def find_faces(self, scope):
        try:
            img = torch.from_numpy(self.video_image).to('cuda')
            img = img.permute(2,0,1)
            kpss = self.detect(img, input_size = (640, 640), max_num=10, metric='default')
            ret = []
            for i in range(kpss.shape[0]):
                if kpss is not None:
                    face_kps = kpss[i]

                face_emb, img_out = self.recognize(img, face_kps)
                ret.append([face_kps, face_emb, img_out])            

        except Exception:
            print(" No media selected")
        
        else:   
            # Find all faces and add to target_faces[]
            if ret:
                # Apply threshold tolerence
                threshhold = self.parameters["ThresholdAmount"][0]/100.0
                
                if self.parameters["ThresholdState"]:
                    threshhold = 0.0           

                # Loop thgouh all faces in video frame
                for face in ret:
                    found = False
                    
                    # Check if this face has already been found
                    for emb in self.target_faces:
                        if self.findCosineDistance(emb['Embedding'], face[1]) < threshhold:
                            found = True
                            break
                    
                    # If we dont find any existing simularities, it means that this is a new face and should be added to our found faces
                    if not found:
                        crop = cv2.resize(face[2].cpu().numpy(), (82, 82))

                        new_target_face = self.target_face.copy()
                        self.target_faces.append(new_target_face)
                        last_index = len(self.target_faces)-1

                        self.target_faces[last_index]["TKButton"] = tk.Button(self.found_faces_canvas, self.inactive_button_style, height = 86, width = 86)
                        self.target_faces[last_index]["TKButton"].bind("<MouseWheel>", self.target_faces_mouse_wheel)
                        self.target_faces[last_index]["ButtonState"] = False           
                        self.target_faces[last_index]["Image"] = ImageTk.PhotoImage(image=Image.fromarray(crop))
                        self.target_faces[last_index]["Embedding"] = face[1]
                        self.target_faces[last_index]["EmbeddingNumber"] = 1
                        
                        # Add image to button
                        self.target_faces[-1]["TKButton"].config( pady = 10, image = self.target_faces[last_index]["Image"], command=lambda k=last_index: self.toggle_found_faces_buttons_state(k))
                        
                        # Add button to canvas
                        self.found_faces_canvas.create_window((last_index)*92, 8, window=self.target_faces[last_index]["TKButton"], anchor='nw') 

                        self.found_faces_canvas.configure(scrollregion = self.found_faces_canvas.bbox("all")) 

    def clear_faces(self):
        self.target_faces = []
        self.found_faces_canvas.delete("all")    
   
    # toggle the target faces button and make assignments        
    def toggle_found_faces_buttons_state(self, button):  
        # Turn all Target faces off
        for i in range(len(self.target_faces)):
            self.target_faces[i]["ButtonState"] = False
            self.target_faces[i]["TKButton"].config(self.inactive_button_style)
        
        # Set only the selected target face to on
        self.target_faces[button]["ButtonState"] = True
        self.target_faces[button]["TKButton"].config(self.button_highlight_style) 

        # set all source face buttons to off
        for i in range(len(self.source_faces)):                
            self.source_faces[i]["ButtonState"] = False
            self.source_faces[i]["TKButton"].config(self.inactive_button_style)
        
        # turn back on the ones that are assigned to the curent target face
        for i in range(len(self.target_faces[button]["SourceFaceAssignments"])):
            self.source_faces[self.target_faces[button]["SourceFaceAssignments"][i]]["ButtonState"] = True
            self.source_faces[self.target_faces[button]["SourceFaceAssignments"][i]]["TKButton"].config(self.button_highlight_style) 

    def toggle_source_faces_buttons_state(self, event, button):  
        # jot down the current state of the button
        state = self.source_faces[button]["ButtonState"]

        # Set all Source Face buttons to False 
        for face in self.source_faces:      
            face["TKButton"].config(self.inactive_button_style)
            face["ButtonState"] = False

        # Toggle the selected Source Face
        self.source_faces[button]["ButtonState"] = not state
        
        # If the source face is now on
        if self.source_faces[button]["ButtonState"]:
            self.source_faces[button]["TKButton"].config(self.button_highlight_style)
        else:
            self.source_faces[button]["TKButton"].config(self.inactive_button_style)

        # Determine which target face is selected
        # If there are target faces
        if self.target_faces:
            for face in self.target_faces:
                
                # Find the first target face that is highlighted
                if face["ButtonState"]:
                    
                    # Clear the assignments
                    face["SourceFaceAssignments"] = []
                    
                    # If a source face is highlighted
                    if self.source_faces[button]["ButtonState"]:
                        # Append new assignment 
                        face["SourceFaceAssignments"].append(button)
                        face['AssignedEmbedding'] = self.source_faces[button]['Embedding']

                    break        

            self.add_action("target_faces", self.target_faces, True, False)

    def toggle_source_faces_buttons_state_shift(self, event, button=-1):  
        # Set all Source Face buttons to False 
        for face in self.source_faces:      
            face["TKButton"].config(self.inactive_button_style)

        # Toggle the selected Source Face
        if button != -1:
            self.source_faces[button]["ButtonState"] = not self.source_faces[button]["ButtonState"]
        
        # Highlight all True buttons
        for face in self.source_faces:  
            if face["ButtonState"]:
                face["TKButton"].config(self.button_highlight_style)

        # If a target face is selected
        for tface in self.target_faces:
            if tface["ButtonState"]:
            
                # Clear all of the assignments
                tface["SourceFaceAssignments"] = []
                tface['AssignedEmbedding'] = np.zeros(512, dtype=np.float32)
                
                # Iterate through all Source faces
                num = 0
                for j in range(len(self.source_faces)):  
                    
                    # If the source face is active
                    if self.source_faces[j]["ButtonState"]:
                        tface["SourceFaceAssignments"].append(j)
                        tface['AssignedEmbedding'] += self.source_faces[j]['Embedding']
                        
                        num +=1
                
                if num>0:
                    tface['AssignedEmbedding'] /= float(num)
                
                break

        self.add_action("target_faces", self.target_faces, True, False)
    
    def populate_target_videos(self):
        # Recursively read all media files from directory
        directory =  self.json_dict["source videos"]
        filenames = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(directory) for f in filenames]         
        
        videos = []
        images = []        
        self.target_media = []
        self.target_media_buttons = []
        self.target_media_canvas.delete("all")  
        
        for file in filenames: # Does not include full path
            # Guess File type based on extension
            try:
                file_type = mimetypes.guess_type(file)[0][:5]
            except:
                print('Unrecognized file type:', file)
            else:
                # Its an image
                if file_type == 'image':
                    try:
                        image = cv2.imread(file)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    except:
                        print('Trouble reading file:', file)
                    else: 
                        ratio = float(image.shape[0]) / image.shape[1]

                        if ratio>1:
                            new_height = 82
                            new_width = int(new_height / ratio)
                        else:
                            new_width = 82
                            new_height = int(new_width * ratio)
                        
                        det_scale = float(new_height) / image.shape[0]
                        image = cv2.resize(image, (new_width, new_height))
                        
                        det_img = np.zeros( (82, 82, 3), dtype=np.uint8 )
                        image[:new_height, :new_width, :] = image
                        images.append([image, file]) 
                
                # Its a video
                elif file_type == 'video':
                    try:
                        video = cv2.VideoCapture(file)
                    except:
                        print('Trouble reading file:', file)
                    else:
                        if video.isOpened(): 
   
                            # Grab a frame from the middle for a thumbnail
                            video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_FRAME_COUNT)/2))
                            success, video_frame = video.read()
                            
                            if success:
                                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)   

                                ratio = float(video_frame.shape[0]) / video_frame.shape[1]

                                if ratio>1:
                                    new_height = 82
                                    new_width = int(new_height / ratio)
                                else:
                                    new_width = 82
                                    new_height = int(new_width * ratio)
                                
                                det_scale = float(new_height) / video_frame.shape[0]
                                video_frame = cv2.resize(video_frame, (new_width, new_height))
                                
                                det_img = np.zeros( (82, 82, 3), dtype=np.uint8 )
                                video_frame[:new_height, :new_width, :] = video_frame

                                videos.append([video_frame, file])
                                video.release()
                            
                            else:
                                print('Trouble reading file:', file)
                        else:
                            print('Trouble opening file:', file)
                


        if self.actions['ImgVidMode'] == 1:
            for i in range(len(images)):
                self.target_media_buttons.append(tk.Button(self.target_media_canvas, self.inactive_button_style, height = 86, width = 86))

                rgb_video = Image.fromarray(images[i][0])        
                self.target_media.append(ImageTk.PhotoImage(image=rgb_video))            
                self.target_media_buttons[i].config( image = self.target_media[i],  command=lambda i=i: self.load_target(i, images[i][1], self.actions['ImgVidModes'][self.actions['ImgVidMode']]))
                self.target_media_buttons[i].bind("<MouseWheel>", self.target_videos_mouse_wheel)
                self.target_media_canvas.create_window(i*92, 8, window = self.target_media_buttons[i], anchor='nw')

            self.target_media_canvas.configure(scrollregion = self.target_media_canvas.bbox("all"))
       
        elif self.actions['ImgVidMode'] == 0:
            for i in range(len(videos)):
                self.target_media_buttons.append(tk.Button(self.target_media_canvas, self.inactive_button_style, height = 86, width = 86))

                rgb_video = Image.fromarray(videos[i][0])        
                self.target_media.append(ImageTk.PhotoImage(image=rgb_video))            
                self.target_media_buttons[i].config( image = self.target_media[i],  command=lambda i=i: self.load_target(i, videos[i][1], self.actions['ImgVidModes'][self.actions['ImgVidMode']]))
                self.target_media_buttons[i].bind("<MouseWheel>", self.target_videos_mouse_wheel)
                self.target_media_canvas.create_window(i*92, 8, window = self.target_media_buttons[i], anchor='nw')

            self.target_media_canvas.configure(scrollregion = self.target_media_canvas.bbox("all"))

    def auto_swap(self):
            # Reselect Target Image
            try:    
                self.find_faces('current')
                self.target_faces[0]["ButtonState"] = True
                self.target_faces[0]["TKButton"].config(self.button_highlight_style) 
                
                # Reselct Source images
                self.toggle_source_faces_buttons_state_shift(None, button=-1)
                
                self.toggle_swapper(True)
            except:
                pass
    def toggle_auto_swap(self):
        self.actions['AutoSwapState'] = not self.actions['AutoSwapState']
        
        if self.actions['AutoSwapState']:
            self.actions['AutoSwapButton'].config(self.active_button_style)
        else:
            self.actions['AutoSwapButton'].config(self.inactive_button_style)
    
    def load_target(self, button, media_file, media_type):
        self.video_loaded = True
        self.clear_faces()
       
        if media_type == 'Videos':
            self.video_slider.set(0)
            self.video_length = []
            self.add_action("load_target_video", media_file, False)
            

        elif media_type == 'Images':
            self.add_action("load_target_image", media_file, False)
            self.image_file_name = os.path.splitext(os.path.basename(media_file))
            
            # # find faces
            if self.actions['AutoSwapState']:
                self.add_action('function', "gui.auto_swap()")

            
        
        self.set_status(media_file) 
        for i in range(len(self.target_media_buttons)):
            self.target_media_buttons[i].config(self.inactive_button_style)
        
        self.target_media_buttons[button].config(self.button_highlight_style)
        
        if self.actions['SwapFacesState'] == True:
            self.toggle_swapper()
        
        if self.play_video == True:
            self.toggle_play_video()

        
        # delete all markers
        for i in range(len(self.markers)):
            self.video_slider_canvas.delete(self.markers[i]['icon_ref'])
        
        self.markers = []
        self.add_action("markers", self.markers)
      
        self.stop_marker = []
        self.video_slider_canvas.delete(self.stop_image)
    
    # @profile
    def set_image(self, image, requested):
        self.video_image = image[0]
        frame = image[1]
        if not requested:
            self.set_slider_position(frame)
            self.parameter_update_from_marker(frame)

        self.resize_image()

    # @profile    
    def resize_image(self):
        image = self.video_image

        if len(image) != 0:
            if self.docked: 
                x1 = float(self.video.winfo_width())
                y1 = float(self.video.winfo_height())

            else:

                x1 = float(self.undock_video.winfo_width())
                y1 = float(self.undock_video.winfo_height())
                if x1==1.0 or y1==1.0:
                    # not directly querying winfo since there ia a small delay to create the window
                    x1 = float(self.json_dict['player_geom'][0])
                    y1 = float(self.json_dict['player_geom'][1])
                    
            x2 = float(image.shape[1])
            y2 = float(image.shape[0])
            

            m1 = x1/y1
            m2 = x2/y2
            
            if m2>m1:
                x2 = x1
                y2 = x1/m2
                image = cv2.resize(image, (int(x2), int(y2)))
                padding = int((y1-y2)/2.0)
                image = cv2.copyMakeBorder( image, padding, padding, 0, 0, cv2.BORDER_CONSTANT)            
            else:
                y2=y1
                x2=y2*m2
                image = cv2.resize(image, (int(x2), int(y2)))
                padding=int((x1-x2)/2.0)
                image = cv2.copyMakeBorder( image, 0, 0, padding, padding, cv2.BORDER_CONSTANT) 

            image = Image.fromarray(image)  
            if self.docked:
                self.undock_video.configure(image='')            
                self.video.image = ImageTk.PhotoImage(image)
                self.video.configure(image=self.video.image)

            else:
                self.video.configure(image='')        
                self.undock_video.image = ImageTk.PhotoImage(image)
                self.undock_video.configure(image=self.undock_video.image)

    def redisplay_markers(self):
        width = self.video_slider_canvas.winfo_width()-30
        for i in range(len(self.markers)):
            position = 15+int(width*self.markers[i]['frame']/self.video_slider.configure('to')[4])
            self.video_slider_canvas.delete(self.markers[i]['icon_ref'])                    
            self.markers[i]['icon_ref'] = self.video_slider_canvas.create_image(position, 30, image=self.marker_icon)
        
        if self.stop_marker:
            self.video_slider_canvas.delete(self.stop_image)
            position = 15+int(width*self.stop_marker/self.video_slider.configure('to')[4]) 
            self.stop_image = self.video_slider_canvas.create_image(position, 30, image=self.stop_marker_icon)  

 
    def check_for_video_resize(self):
        if self.docked:
            
            # Read the geometry from the last time json was updated. json only updates once the window ahs stopped changing
            win_geom = '%dx%d+%d+%d' % (self.json_dict['dock_win_geom'][0], self.json_dict['dock_win_geom'][1] , self.json_dict['dock_win_geom'][2], self.json_dict['dock_win_geom'][3])
           
           # window has started changing
            if self.winfo_geometry() != win_geom:
                # Resize image in video window
                self.resize_image()
                    
                # redisplay markers    
                self.redisplay_markers()
                
                # Check if window has stopped changing
                if self.winfo_geometry() != self.window_last_change:
                    self.window_last_change = self.winfo_geometry()

                # The window has stopped changing
                else:
                    # Update json
                    str1 = self.winfo_geometry().split('x')
                    str2 = str1[1].split('+')
                    win_geom = [str1[0], str2[0], str2[1], str2[2]]
                    win_geom = [int(strings) for strings in win_geom]
                    self.json_dict['dock_win_geom'] = win_geom
                    with open("data.json", "w") as outfile:
                        json.dump(self.json_dict, outfile)            

        else:
            # Control window
            # Read the geometry from the last time json was updated. json only updates once the window ahs stopped changing
            win_geom = '%dx%d+%d+%d' % (self.json_dict['undock_win_geom'][0], self.json_dict['undock_win_geom'][1] , self.json_dict['undock_win_geom'][2], self.json_dict['undock_win_geom'][3])
            
            # window has started changing
            if self.winfo_geometry() != win_geom:
                # redisplay markers      
                self.redisplay_markers()          
                
                # Check if window has stopped changing
                if self.winfo_geometry() != self.window_last_change:
                    self.window_last_change = self.winfo_geometry()
                    
                # The window has stopped changing
                else:
                    # update json
                    str1 = self.winfo_geometry().split('x')
                    str2 = str1[1].split('+')
                    win_geom = [str1[0], str2[0], str2[1], str2[2]]
                    win_geom = [int(strings) for strings in win_geom]
                    self.json_dict['undock_win_geom'] = win_geom
                    with open("data.json", "w") as outfile:
                        json.dump(self.json_dict, outfile)   

            # Preview window 
            # Read the geometry from the last time json was updated. json only updates once the window ahs stopped changing
            win_geom = '%dx%d+%d+%d' % (self.json_dict['player_geom'][0], self.json_dict['player_geom'][1] , self.json_dict['player_geom'][2], self.json_dict['player_geom'][3])
            
            # window has started changing
            if self.undock_video_window.winfo_geometry() != win_geom:
                # Resize image in video window
                self.resize_image()
            
                # Check if window has stopped changing
                if self.undock_video_window.winfo_geometry() != self.undock_video_last_change:            
                    self.undock_video_last_change = self.undock_video_window.winfo_geometry()
                    
                # The window has stopped changing
                else:
                    # update json
                    str1 = self.undock_video_window.winfo_geometry().split('x')
                    str2 = str1[1].split('+')
                    win_geom = [str1[0], str2[0], str2[1], str2[2]]
                    win_geom = [int(strings) for strings in win_geom]
                    self.json_dict['player_geom'] = win_geom
                    with open("data.json", "w") as outfile:
                        json.dump(self.json_dict, outfile) 
       
    def get_action(self):
        action = self.action_q[0]
        self.action_q.pop(0)
        return action
        
    def get_action_length(self):
        return len(self.action_q)
      

        
    def set_video_slider_length(self, video_length):
        self.video_length = video_length
        self.video_slider.configure(to=video_length)

    def set_slider_position(self, position):
        self.video_slider.set(position)
   
    def findCosineDistance(self, vector1, vector2):

        return 1 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))


    def toggle_play_video(self):
        if self.actions['ImgVidMode'] == 0:
            if not self.video_loaded:
                print("Please select video first!")
                return
            self.play_video = not self.play_video
            
            if self.play_video:
                if self.rec_video: 
                    if not self.json_dict["saved videos"]:
                        print("Set saved video folder first!")
                        self.play_video = False
                        self.add_action("play_video", "stop")
                        self.actions['PlayButton'].config(self.inactive_button_style)
                    else:
                        self.add_action("play_video", "record")
                        self.actions['PlayButton'].config(self.active_button_style)
                else:
                    self.add_action("play_video", "play")
                    self.actions['PlayButton'].config(self.active_button_style)
                
            else:
                self.add_action("play_video", "stop")
                self.actions['PlayButton'].config(self.inactive_button_style)
                if self.rec_video:
                    self.toggle_rec_video()

    def set_player_buttons_to_inactive(self):
        self.actions['PlayButton'].config(self.inactive_button_style)
        self.actions['RecordButton'].config(self.inactive_button_style)
        self.rec_video = False
        self.play_video = False
    
    
    def toggle_swapper(self, toggle_value=-1):
        if toggle_value == -1:
            self.actions['SwapFacesState'] = not self.actions['SwapFacesState']
        else:
            self.actions['SwapFacesState'] = toggle_value
        
        if not self.actions['SwapFacesState']:
            self.actions['SwapFacesButton'].config(self.inactive_button_style)
        else:
            self.actions['SwapFacesButton'].config(self.active_button_style) 

        self.add_action("swap", self.actions['SwapFacesState'], True)

            
    def toggle_rec_video(self):
        if not self.play_video:
            self.rec_video = not self.rec_video
                
            if self.rec_video == False:
                self.actions['RecordButton'].config(self.inactive_button_style)
            else:
                self.actions['RecordButton'].config(self.active_button_style, bg='red') 

 
    def update_CLIP_text(self, text):
        self.parameters['CLIPText'] = text.get()
        self.add_action("parameters", self.parameters, True)
        self.focus()
    
    def add_action(self, action, parameter=None, request_updated_frame=False, ignore_markers=True):
        
        self.action_q.append([action, parameter]) 
        
        if not self.play_video and request_updated_frame:

            if self.actions['ImgVidMode'] == 0 and not ignore_markers:
                self.action_q.append(["get_requested_video_frame", self.video_slider.get()])
            else:
                self.action_q.append(["get_requested_video_frame_parameters", self.video_slider.get()])

    def toggle_dock(self):
        if self.docked:
            self.docked = False
            
            # make pop up window visible    
            self.undock_video_window.deiconify()
            self.undock_video_window.geometry('%dx%d+%d+%d' % (self.json_dict['player_geom'][0], self.json_dict['player_geom'][1] , self.json_dict['player_geom'][2], self.json_dict['player_geom'][3]))  
            
            # resize mainwindow
            self.grid_rowconfigure(0, weight = 0) 
            self.geometry('%dx%d+%d+%d' % (self.json_dict['undock_win_geom'][0], self.json_dict['undock_win_geom'][1] , self.json_dict['undock_win_geom'][2], self.json_dict['undock_win_geom'][3]))            
            self.resizable(width=True, height=False) 
            
            self.resize_image()
            self.redisplay_markers()

        else:
            self.docked = True
            
            # hide pop up window
            self.undock_video_window.withdraw() 
            
            # reshow video in main
            self.grid_rowconfigure(0, weight = 10)           
            self.geometry('%dx%d+%d+%d' % (self.json_dict['dock_win_geom'][0], self.json_dict['dock_win_geom'][1] , self.json_dict['dock_win_geom'][2], self.json_dict['dock_win_geom'][3])) 
            self.resizable(width=True, height=True) 
            self.resize_image()
        
    def set_status(self, msg):
        self.status_label.configure(text=str(msg))
        # self.status_label.pack()
        
    def mouse_wheel(self, event, frame):    
        if event.delta > 0: 
            frame += 1
        else:
            frame -= 1

        self.video_slider.set(frame)
        self.add_action("get_requested_video_frame", frame)
        
        self.parameter_update_from_marker(frame)
            
    def save_selected_source_faces(self, text):        
        # get name from text field
        text = text.get()
        # get embeddings from all highlightebuttons
        # iterate through the buttons
        summed_embeddings = [0]*512
        num_selected = 0
        for i in range(len(self.source_faces)):
            if self.source_faces[i]["ButtonState"]:
                summed_embeddings += self.source_faces[i]["Embedding"]
                num_selected += 1
        
        # create the average embedding
        if num_selected != 0:
            ave_embedding = summed_embeddings/num_selected
            
            if text != "":
                with open("merged_embeddings.txt", "a") as embedfile:
                    identifier = "Name: "+text
                    embedfile.write("%s\n" % identifier)
                    for number in ave_embedding:
                        embedfile.write("%s\n" % number)
            else:
                print('No embedding name specified')
        else:
            print('No Source Images selected')
        
        self.focus()
        self.load_source_faces()
    

    def delete_merged_embedding(self): #add multi select
    
    # get selected button
        sel = []
        for j in range(len(self.source_faces)):
            if self.source_faces[j]["ButtonState"]:
                sel = j
                break
        
        # check if it is a merged embedding
        # if so, read txt embedding into list
        temp0 = []
        if os.path.exists("merged_embeddings.txt"):

            with open("merged_embeddings.txt", "r") as embedfile:
                temp = embedfile.read().splitlines() 

                for i in range(0, len(temp), 513):
                    to = [temp[i], np.array(temp[i+1:i+513], dtype='float32')]
                    temp0.append(to)  
                  
        if j < len(temp0):
            temp0.pop(j)
        
            with open("merged_embeddings.txt", "w") as embedfile:
                for line in temp0:                    
                    embedfile.write("%s\n" % line[0])
                    for i in range(512):
                        embedfile.write("%s\n" % line[1][i])
    
        self.load_source_faces()
        
    def iterate_through_merged_embeddings(self, event):
        if event.delta>0:
            for i in range(len(self.source_faces)):  
                if self.source_faces[i]["ButtonState"] and i<len(self.source_faces)-1:
                    self.toggle_source_faces_buttons_state(None, i+1)
                    break
        elif event.delta<0:
            for i in range(len(self.source_faces)):  
                if self.source_faces[i]["ButtonState"]and i>0:
                    self.toggle_source_faces_buttons_state(None, i-1)
                    break
        
    def toggle_ui_button(self, parameter, toggle_value=-1):
        state = parameter+'State'
        button = parameter+'Button'
        
        if toggle_value == -1:
            self.parameters[state] = not self.parameters[state]
        else:
            self.parameters[state] = toggle_value
        
        if self.parameters[state]:
            self.param_const[button].config(self.active_button_style)
        else:
            self.param_const[button].config(self.inactive_button_style)
            
        self.add_action("parameters", self.parameters, True)   
        
 
 
    
    # update ui button states
    def set_parameter(self, parameter, value):
        self.parameters[parameter] = value
        
        if self.parameters[parameter]:
            self.parameters_buttons[parameter].config(self.active_button_style)
        else:
            self.parameters_buttons[parameter].config(self.inactive_button_style)
        
    
    def parameter_amount(self, event, parameter, parameter_amount, increment, maximum, minimum=0, unit='%' ):
        if parameter_amount != '':        
            self.parameters[parameter_amount] += increment*int(event.delta/120.0)
            if self.parameters[parameter_amount] > maximum:
                self.parameters[parameter_amount] = maximum
            if self.parameters[parameter_amount] < minimum :
                self.parameters[parameter_amount] = minimum

            temp_num = str(self.parameters[parameter_amount])
            temp = ' '+parameter+' '*(12-len(parameter)-len(temp_num))+temp_num+unit
        else:
            temp = ' '+parameter
        
        self.parameters_buttons[parameter].config(text=temp)        
        self.add_action("parameters", self.parameters, True)   

    
 
    
    # update ui button values
    def set_parameter_amount(self, parameter, parameter_amount, value, maximum, unit='%'):
        self.parameters[parameter_amount] = value

        temp_num = str(self.parameters[parameter_amount])
        temp = ' '+parameter+' '*(12-len(parameter)-len(temp_num))+temp_num+unit

        self.parameters_buttons[parameter].config(text=temp)        
          

    def change_video_quality(self, event): 
        self.video_quality += (1*int(event.delta/120.0))
        
        if self.video_quality > 50:
            self.video_quality = 50
        if self.video_quality < 0 :
            self.video_quality = 0
        
        parameter = 'Vid Qual'
        temp_num = str(int(self.video_quality))
        temp = ' '+parameter+' '*(13-len(parameter)-len(temp_num))+temp_num

        self.actions['VideoQualityButton'].config(text=temp)        
 
        self.add_action("vid_qual",int(self.video_quality))

    def change_threads_amount(self, event): 
        self.num_threads += (1*int(event.delta/120.0))
        
        if self.num_threads > 10:
            self.num_threads = 10
        if self.num_threads < 1:
            self.num_threads = 1
        
        parameter = 'Threads'
        temp_num = str(int(self.num_threads))
        temp = ' '+parameter+' '*(13-len(parameter)-len(temp_num))+temp_num
            
        self.actions['ThreadsButton'].config(text=temp)        

        self.add_action("num_threads",int(self.num_threads))
        
        self.json_dict["threads"] = self.num_threads
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)

    def slider_move(self, button_state, current_frame):
        global last_frame
        
        if button_state == 'motion' and current_frame != last_frame:
            self.add_action("get_requested_video_frame", current_frame)
            last_frame = current_frame

        elif button_state == 'press':
            self.add_action("swap", False, True)
                
        elif button_state == 'release':
            self.add_action("swap", self.actions['SwapFacesState'], True, False)
            
        self.parameter_update_from_marker(current_frame)
            



    def toggle_vid_img(self):
        if self.actions['ImgVidMode'] == 0:
            self.actions['ImgVidMode'] = 1
            self.media_control_canvas.grid_remove()
            self.image_control_canvas.grid()
        else:
            self.actions['ImgVidMode'] = 0
            self.image_control_canvas.grid_remove()
            self.media_control_canvas.grid()
        
        index = self.actions['ImgVidMode']
        mode = self.actions['ImgVidModes'][index]
        
        temp = ' '+mode

        self.actions['ImgVidButton'].config(text=temp) 
        self.populate_target_videos()
        
        self.add_action("parameters", self.parameters)  
        # self.add_action('load_null')
        
        # Reset relavent GUI
        if self.actions['SwapFacesState'] == True:
            self.toggle_swapper()
        
        if self.play_video == True:
            self.toggle_play_video()
        
        self.clear_faces()
        self.video_loaded = False
        # delete all markers
        for i in range(len(self.markers)):
            self.video_slider_canvas.delete(self.markers[i]['icon_ref'])
        
        self.markers = []
        self.add_action("markers", self.markers)   


    def toggle_perf_test(self):
        self.actions['PerfTestState'] = not self.actions['PerfTestState']
        
        if self.actions['PerfTestState']:
            self.actions['PerfTestButton'].config(self.active_button_style)
        else:
            self.actions['PerfTestButton'].config(self.inactive_button_style)

        self.add_action('perf_test', self.actions['PerfTestState']) 
        
        
    def add_marker(self):
         # Delete existing marker at current frame and replace with new data
        for i in range(len(self.markers)):
            if self.markers[i]['frame'] == self.video_slider.get():
                self.video_slider_canvas.delete(self.markers[i]['icon_ref'])
                self.markers.pop(i)
                break

        width = self.video_slider_canvas.winfo_width()-30
        position = 15+int(width*self.video_slider.get()/self.video_slider.configure('to')[4])

        temp_param = copy.deepcopy(self.parameters)
        
        temp = {
                'frame':        self.video_slider.get(),
                'parameters':   temp_param,
                'icon_ref':     self.video_slider_canvas.create_image(position, 30, image=self.marker_icon),
                }

        self.markers.append(temp)

        def sort(e):
            return e['frame']    
        
        self.markers.sort(key=sort)

        self.add_action("markers", self.markers)

    def next_marker(self):
        temp=[]
        for i in range(len(self.markers)):
            temp.append(self.markers[i]['frame'])
        idx = bisect.bisect(temp, self.video_slider.get())
        
        if idx < len(self.markers):
            self.video_slider.set(self.markers[idx]['frame'])
            
            # self.add_action("get_requested_video_frame", self.markers[idx]['frame'])  
            self.add_action('get_requested_video_frame', self.markers[idx]['frame'])
            self.parameter_update_from_marker(self.markers[idx]['frame'])
        
    def previous_marker(self):
        temp=[]
        for i in range(len(self.markers)):
            temp.append(self.markers[i]['frame'])
        idx = bisect.bisect_left(temp, self.video_slider.get())
        
        if idx > 0:
        
            self.video_slider.set(self.markers[idx-1]['frame'])
            
            # self.add_action("get_requested_video_frame", self.markers[idx-1]['frame'])  
            self.add_action('get_requested_video_frame', self.markers[idx-1]['frame'])
            self.parameter_update_from_marker(self.markers[idx-1]['frame'])

    def remove_marker(self):
        for i in range(len(self.markers)):
            if self.markers[i]['frame'] == self.video_slider.get():
                self.video_slider_canvas.delete(self.markers[i]['icon_ref'])
                self.markers.pop(i)
                break
                
    def toggle_stop(self):
        if self.stop_marker == self.video_slider.get():
            self.stop_marker = []
            self.add_action('set_stop', -1)
            self.video_slider_canvas.delete(self.stop_image)
        else:
            self.video_slider_canvas.delete(self.stop_image)
            self.stop_marker = self.video_slider.get()
            self.add_action('set_stop', self.stop_marker)
        
            width = self.video_slider_canvas.winfo_width()-30
            position = 15+int(width*self.video_slider.get()/self.video_slider.configure('to')[4])  
            self.stop_image = self.video_slider_canvas.create_image(position, 30, image=self.stop_marker_icon)

  
    def save_image(self):
        filename =  self.image_file_name[0]+"_"+str(time.time())[:10]
        filename = os.path.join(self.json_dict["saved videos"], filename)
        cv2.imwrite(filename+'.jpg', cv2.cvtColor(self.video_image, cv2.COLOR_BGR2RGB))
    

    def create_ui_button(self, parameter, root, x, y, width=125, height=26):
        icon = parameter+'Icon'
        icon_holder = parameter+'IconHolder'
        button = parameter+'Button'
        message = parameter+'Message'

        # Add Icon
        img = Image.open(self.parameters[icon])
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.param_const[icon_holder] = ImageTk.PhotoImage(resized_image)
        
        # Create Button and place
        # L-Click function - On/off
        self.param_const[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.param_const[icon_holder], anchor='w', command=lambda: self.toggle_ui_button(parameter))        
        self.param_const[button].place(x=x, y=y, width=width, height=height) 
        
        # R-click function - Cycle modes
        self.param_const[button].bind("<ButtonRelease-3>", lambda event:self.cycle_ui_button_modes(parameter))
        
        # Mousewheel function - adjust parameter
        self.param_const[button].bind("<MouseWheel>", lambda event: self.update_parameter_data(event, parameter))
        
        # Status Text
        self.param_const[button].bind('<Enter>', lambda event: self.set_status(self.parameters[message]))
        self.param_const[button].bind('<Leave>', lambda event: self.set_status(''))

    def cycle_ui_button_modes(self, parameter):
        index = self.parameters[parameter+'Mode']
        mode = self.parameters[parameter+'Modes'][index]
        
        index += 1
        
        if index > len(self.parameters[parameter+'Modes'])-1:
            index=0
            
        self.parameters[parameter+'Mode'] = index
        
        self.update_ui_button(parameter)
        self.add_action('parameters', self.parameters, True)
        
    def update_parameter_data(self, event, parameter):

    
        amount = parameter+'Amount'
        index = self.parameters[parameter+'Mode']
        minimum = self.parameters[parameter+'Min']
        maximum = self.parameters[parameter+'Max']
        increment = self.parameters[parameter+'Inc']
  
        self.parameters[amount][index] += increment*int(event.delta/120.0)
        if self.parameters[amount][index] > maximum:
            self.parameters[amount][index] = maximum
        if self.parameters[amount][index] < minimum:
            self.parameters[amount][index] = minimum
        


            
        self.update_ui_button(parameter)
        self.add_action("parameters", self.parameters, True)   


  
 
    # update ui button values
    def update_ui_button(self, parameter):
        index = self.parameters[parameter+'Mode']
        amount = self.parameters[parameter+'Amount'][index]
        unit = self.parameters[parameter+'Unit']
        index = self.parameters[parameter+'Mode']
        mode = self.parameters[parameter+'Modes'][index]
        increment = self.parameters[parameter+'Inc']
        button = parameter+'Button'
        state = self.parameters[parameter+'State']
        
        if increment != 0:
            temp_num = str(amount)
            temp = ' '+mode+' '*(12-len(mode)-len(temp_num))+temp_num+unit
        else:
            temp = ' '+mode

        self.param_const[button].config(text=temp)  
        
        if state:
            self.param_const[button].config(self.active_button_style)
        else: 
            self.param_const[button].config(self.inactive_button_style)
        
    def create_ui_button_2(self, parameter, root, x, y, width=96, height=26):
        icon = parameter+'Icon'
        icon_holder = parameter+'IconHolder'
        button = parameter+'Button'
        message = parameter+'Message'
        index = self.actions[parameter+'Mode']
        mode = self.actions[parameter+'Modes'][index]

        # Add Icon
        img = Image.open(self.actions[icon])
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.actions[icon_holder] = ImageTk.PhotoImage(resized_image)
        
        # Create Button and place
        # L-Click function - On/off
        if parameter == 'Dock':
            resized_image= img.resize((12,30), Image.ANTIALIAS)
            self.actions[icon_holder] = ImageTk.PhotoImage(resized_image)
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.toggle_dock())
            
        if parameter == 'ImgDock':
            resized_image= img.resize((12,30), Image.ANTIALIAS)
            self.actions[icon_holder] = ImageTk.PhotoImage(resized_image)
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.toggle_dock())            
        
        elif parameter == 'SaveImage':   
            resized_image= img.resize((30,30), Image.ANTIALIAS)
            self.actions[icon_holder] = ImageTk.PhotoImage(resized_image)
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.save_image())    
        elif parameter == 'AutoSwap':   
            resized_image= img.resize((30,30), Image.ANTIALIAS)
            self.actions[icon_holder] = ImageTk.PhotoImage(resized_image)
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.toggle_auto_swap())               
       
        elif parameter == 'Play':
            resized_image= img.resize((30,30), Image.ANTIALIAS)
            self.actions[icon_holder] = ImageTk.PhotoImage(resized_image)
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.toggle_play_video())
        
        elif parameter == 'Record':   
            resized_image= img.resize((30,30), Image.ANTIALIAS)
            self.actions[icon_holder] = ImageTk.PhotoImage(resized_image)
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.toggle_rec_video())

        elif parameter == 'AddMarker':
            resized_image= img.resize((15,30), Image.ANTIALIAS)
            self.actions[icon_holder] = ImageTk.PhotoImage(resized_image)
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.add_marker())
        elif parameter == 'RemoveMarker':   
            resized_image= img.resize((15,30), Image.ANTIALIAS)
            self.actions[icon_holder] = ImageTk.PhotoImage(resized_image)
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.remove_marker())
        elif parameter == 'PrevMarker':   
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.previous_marker())
        elif parameter == 'NextMarker':   
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.next_marker())        
        elif parameter == 'ToggleStop':   
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.toggle_stop())               

            
        elif parameter == 'FindFaces':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.add_action("find_faces", "current"))    
       
        elif parameter == 'ClearFaces':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.add_action("clear_faces", "current"))  

        elif parameter == 'SwapFaces':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.toggle_swapper())
        
        
        elif parameter == 'LoadSFaces':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.select_faces_path())
        elif parameter == 'DelEmbed':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.delete_merged_embedding())
        elif parameter == 'LoadTVideos':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.select_video_path())
        elif parameter == 'ImgVid':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.toggle_vid_img())
        # elif parameter == 'HoldFace':
            # self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', text=self.actions['HoldFaceModes'][0], image=self.actions[icon_holder], anchor='w', command=lambda: self.toggle_hold_face())
            
        elif parameter == 'StartRope':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.load_all())
        elif parameter == 'OutputFolder':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.select_save_video_path())
        elif parameter == 'Threads':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w')
            self.actions[button].bind("<MouseWheel>", self.change_threads_amount)  
        # elif parameter == 'VideoQuality':
            # self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w')
            # self.actions[button].bind("<MouseWheel>", self.change_video_quality) 
        elif parameter == 'PerfTest':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', text=self.actions['PerfTestModes'][0], image=self.actions[icon_holder], anchor='w', command=lambda: self.toggle_perf_test())
        elif parameter == 'Clearmem':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', text=self.actions['ClearmemModes'][0], image=self.actions[icon_holder], anchor='w', command=lambda: self.clear_mem())            

            
        temp = ' '+mode

        self.actions[button].config(text=temp) 
        self.actions[button].place(x=x, y=y, width=width, height=height)        

        # Status Text
        self.actions[button].bind('<Enter>', lambda event: self.set_status(self.actions[message]))
        self.actions[button].bind('<Leave>', lambda event: self.set_status(''))
        
    def clear_mem(self):
        self.toggle_swapper(False)
        self.toggle_ui_button('Upscale', False)
        self.toggle_ui_button('Upscale', False)
        self.toggle_ui_button('CLIP', False)
        self.toggle_ui_button('Occluder', False)
        self.toggle_ui_button('FaceParser', False)
        self.add_action('clear_mem', None)
        
    def parameter_update_from_marker(self, frame):
    
        # sync marker data
        temp=[]
        # create a separate list with the list of frame numbers with markers
        for i in range(len(self.markers)):
            temp.append(self.markers[i]['frame'])
        # find the marker frame to the left of the current frame
        idx = bisect.bisect(temp, frame) 
        # update UI with current marker state data
        if idx>0:
            # update paramter dict with marker entry 
            self.parameters = copy.deepcopy(self.markers[idx-1]['parameters'])

            # update buttons
            self.update_ui_button('Upscale')
            self.update_ui_button('Diff')
            self.update_ui_button('Border')
            self.update_ui_button('MaskView')
            self.update_ui_button('CLIP')
            self.update_ui_button('Occluder')
            self.update_ui_button('FaceParser')
            self.update_ui_button('Blur')
            self.update_ui_button('Threshold')
            self.update_ui_button('Strength')
            self.update_ui_button('Orientation')
            
            self.CLIP_text.delete(0, tk.END)
            self.CLIP_text.insert(0, self.parameters['CLIPText'])

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

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
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
        syncvec = torch.empty((1,1), dtype=torch.float32, device='cuda:0')
        syncvec = syncvec.cpu()        
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
        img_out = img
        
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
        syncvec = torch.empty((1,1), dtype=torch.float32, device='cuda:0')
        syncvec = syncvec.cpu()
        self.recognition_model.run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), img_out               

    

        
        