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

from  rope.Dicts import PARAM_BUTTONS_PARAMS, ACTIONS, PARAM_BUTTONS_CONSTANT

last_frame = 0

class GUI(tk.Tk):
    def __init__( self ):  
        super().__init__()
        # Adding a title to the self
        # self.call('tk', 'scaling', 0.5)
        self.title("Test Application")
        self.pixel = []
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

        self.parameters = PARAM_BUTTONS_PARAMS
        self.actions = ACTIONS
        self.param_const = PARAM_BUTTONS_CONSTANT


        self.parameters_buttons={}
                 
        self.icons =   {
                            "GFPGAN":                   [],
                            "Diff":                     [],
                            "Threshold":                [],
                            "MaskTop":                  [],
                            "MaskBlur":                 [],
                            "Occluder":                 [],
                            "CLIP":                     [],
                            'FindFaces':                [],
                            }
        
        self.button_data =  {
                            'FindFaces':                ['./rope/media/tarface.png', '', ''],
                            }
                            
       
        
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
        self.faceapp_model = []

        self.video_loaded = False

        self.dock = True
        self.undock = []
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)   
        self.image_file_name = []

      
        self.json_dict = {"source videos":None, "source faces":None, "saved videos":None, "threads":1}
        
        self.new_int = tk.IntVar()
        self.marker =  {
                        'frame':        '0',
                        'parameters':   '',
                        'icon_ref':     '',
                        }
        self.markers = []                
        
        self.button1 = "gray25"
        self.button_1_text = "light goldenrod"
        self.button1_active = "gray50"
        
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
                                                    
        # Media frame
        self.video_frame = tk.Frame( self, self.frame_style)
        self.video_frame.grid( row = 0, column = 0, sticky='NEWS', pady = 2 )
        
        self.video_frame.grid_columnconfigure(0, weight=1)  
        self.video_frame.grid_rowconfigure(0, weight = 1)
    
        # Video [0,0]
        self.video = tk.Label( self.video_frame, self.label_style, bg='black')
        self.video.grid( row = 0, column = 0, sticky='NEWS', pady =0 )
        self.video.bind("<MouseWheel>", self.iterate_through_merged_embeddings)
        
        # Media control canvas
        self.media_control_canvas = tk.Canvas( self.video_frame, self.canvas_style1, height = 40)
        self.media_control_canvas.grid( row = 1, column = 0, sticky='NEWS', pady = 0)  
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

        img = Image.open('./rope/media/marker.png')
        resized_image= img.resize((15,30), Image.ANTIALIAS)
        self.marker_icon = ImageTk.PhotoImage(resized_image)    

        # Marker canvas
        self.marker_canvas = tk.Canvas( self.media_control_canvas, self.canvas_style1, width = 150, height = 40)
        self.marker_canvas.grid( row = 0, column = 2, sticky='news', pady = 0)        
        
        # Marker Buttons
        self.create_ui_button_2('AddMarker', self.marker_canvas, 8, 2, width=36, height=36)
        self.create_ui_button_2('RemoveMarker', self.marker_canvas, 35, 2, width=36, height=36)
        self.create_ui_button_2('PrevMarker', self.marker_canvas, 69, 2, width=36, height=36)
        self.create_ui_button_2('NextMarker', self.marker_canvas, 107, 2, width=36, height=36)

        # Image control canvas
        self.image_control_canvas = tk.Canvas( self.video_frame, self.canvas_style1, height = 40)
        self.image_control_canvas.grid( row = 1, column = 0, sticky='NEWS', pady = 0)  
        self.image_control_canvas.grid_columnconfigure(1, weight = 1)        
        
        # Image Save
        self.create_ui_button_2('SaveImage', self.image_control_canvas, 31, 2, width=36, height=36)

        
        
 ######### Options
        x_space = 40
        self.options_frame = tk.Frame( self, self.frame_style, height = 71)
        self.options_frame.grid( row = 1, column = 0, sticky='NEWS', pady = 2 )
        
        self.options_frame.grid_rowconfigure( 0, weight = 100 )          
        self.options_frame.grid_columnconfigure( 0, weight = 1 ) 
        
        # Left Canvas
        self.options_frame_canvas1 = tk.Canvas( self.options_frame, self.canvas_style1, height = 71)
        self.options_frame_canvas1.grid( row = 0, column = 0, sticky='NEWS', pady = 0 )

        # Label Frame 1
        self.label_frame1 = tk.LabelFrame( self.options_frame_canvas1, self.frame_style, height = 71, width = 1000 )
        self.label_frame1.place(x=0, y=0)
        
        column1=8
        self.create_ui_button('Upscale', self.label_frame1, column1, 8)
        self.create_ui_button('Diff', self.label_frame1, column1, 37)

        column2=column1+125+x_space
        self.create_ui_button('Mask', self.label_frame1, column2, 8)
        self.create_ui_button('MaskView', self.label_frame1, column2, 37)

        column3=column2+125+x_space
        self.create_ui_button('CLIP', self.label_frame1, column3, 8)

        # CLIP-entry
        self.temptkstr = tk.StringVar(value="")
        self.CLIP_text = tk.Entry(self.label_frame1, relief='flat', bd=0, textvariable=self.temptkstr)
        self.CLIP_text.place(x=column3, y=40, width = 125, height=20) 
        self.CLIP_text.bind("<Return>", lambda event: self.update_CLIP_text())

        column4=column3+125+x_space  
        self.create_ui_button('Occluder', self.label_frame1, column4, 8)
        self.create_ui_button('FaceParser', self.label_frame1, column4, 37)

        column5=column4+125+x_space        
        self.create_ui_button('Blur', self.label_frame1, column5, 8)        
        self.create_ui_button('Threshold', self.label_frame1, column5, 37)
 

        column6=column5+125+x_space
        self.create_ui_button('Strength', self.label_frame1, column6, 8)
        self.create_ui_button('Orientation', self.label_frame1, column6, 37)
        

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
        column=column+125+x_space
        self.create_ui_button_2('VideoQuality', self.program_options_label, column, 8,width = 125, height = 26)
        
        # Status
        self.status_frame = tk.Frame( self, bg='grey20', height = 15)
        self.status_frame.grid( row = 6, column = 0, sticky='NEWS', pady = 2 )
        
        self.status_label = tk.Label(self.status_frame, fg="white", bg='grey20')
        self.status_label.pack()

 
    def target_faces_mouse_wheel(self, event):
        self.found_faces_canvas.xview_scroll(1*int(event.delta/120.0), "units") 
   

    def source_faces_mouse_wheel(self, event):
        self.source_faces_canvas.xview_scroll(1*int(event.delta/120.0), "units")

   
    def target_videos_mouse_wheel(self, event):
        self.target_media_canvas.xview_scroll(1*int(event.delta/120.0), "units")


    def initialize_gui( self ):

        self.title("Rope")
        # self.overrideredirect(True)
        self.configure(bg='grey10')
        self.resizable(width=True, height=True) 

        self.geometry('%dx%d+%d+%d' % (980, 1020, self.winfo_screenwidth()/2-400, self.winfo_screenheight()/2-510))

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
        
        self.actions['StartRopeButton'].configure(self.button_highlight_style, text=' Load Rope')
        
        class empty:
            def __init__(self):
                self.delta = 0
        event = empty()    

        self.update_ui_button('Upscale')
        self.update_ui_button('Diff')
        self.update_ui_button('Mask')
        self.update_ui_button('MaskView')
        self.update_ui_button('CLIP')
        self.update_ui_button('Occluder')
        self.update_ui_button('FaceParser')
        self.update_ui_button('Blur')
        self.update_ui_button('Threshold')
        self.update_ui_button('Strength')
        self.update_ui_button('Orientation')

        self.change_video_quality(event)
        self.change_threads_amount(event)    
        
        self.add_action("parameters", self.parameters)   
        self.set_status('Welcome to Rope-Crystal!')
            
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
        if not self.faceapp_model:
            self.add_action('load_faceapp_model')
        
        else:
            self.source_faces = []
            self.source_faces_canvas.delete("all")

            # First load merged embeddings
            if os.path.exists("merged_embeddings.txt"):

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

            directory = self.json_dict["source faces"]

            if directory == None:
                print("No directory assigned")
            else:
            
                filenames = os.listdir(directory)
                
                faces = []
                
                # Find all faces and ad to faces[]
                for name in filenames: #should check if is an image
                    temp_file = os.path.join(directory, name)
                    temp_file = cv2.imread(temp_file)
                    ret = self.faceapp_model.get(temp_file, max_num=1)
                    if ret:
                        # 128 transforms
                        ratio = 1.0
                        diff_x = 8.0*ratio
                        dst = self.arcface_dst * ratio
                        dst[:,0] += diff_x
                        tform = trans.SimilarityTransform()
                        tform.estimate(ret[0].kps, dst)
                        M128 = tform.params[0:2, :]  
                        IM128 = cv2.invertAffineTransform(M128)
                        
                        orig_bbox = cv2.transform(np.array([[[0,0], [0,128], [128,0], [128,128]]]), np.array(IM128))                    
                    
                        left = floor(min(orig_bbox[0][0][0], orig_bbox[0][1][0] ))
                        if left<0:
                            left=0
                        top = floor(min(orig_bbox[0][0][1], orig_bbox[0][2][1] ))
                        if top<0: 
                            top=0
                        right = ceil(max(orig_bbox[0][2][0], orig_bbox[0][3][0] ))
                        if right>temp_file.shape[1]:
                            right=temp_file.shape[1]        
                        bottom = ceil(max(orig_bbox[0][1][1], orig_bbox[0][3][1] ))
                        if bottom>temp_file.shape[0]:
                            bottom=temp_file.shape[0]

                        y_diff = bottom-top
                        x_diff = right-left
                    
                        crop = temp_file[int(top):int(bottom),int(left):int(right)]#y,x
                        if y_diff > x_diff:
                            padding = int((y_diff - x_diff) / 2)
                            crop = cv2.copyMakeBorder( crop, 0, 0, padding, padding, cv2.BORDER_CONSTANT)
                        else:
                            padding = int((x_diff - y_diff) / 2)
                            crop = cv2.copyMakeBorder( crop, padding, padding, 0, 0, cv2.BORDER_CONSTANT )
                                    
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)            
                        crop = cv2.resize( crop, (82, 82))
                        temp = [crop, ret[0].embedding]
                        faces.append(temp)
                
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
                
                # send over source faces embeddings
                self.add_action("source_embeddings", self.source_faces)
        
    def find_faces(self, scope):
        try:
            ret = self.faceapp_model.get(self.video_image, max_num=10)
        except Exception:
            print(" No video selected")
        else:    
            # Find all faces and add to faces[]
            if ret:
                # Loop thgouh all faces in video frame
                for i in range(len(ret)):
                    # Create a frame for each face
                    # 128 transforms
                    ratio = 1.0
                    diff_x = 8.0*ratio
                    dst = self.arcface_dst * ratio
                    dst[:,0] += diff_x
                    tform = trans.SimilarityTransform()
                    tform.estimate(ret[i].kps, dst)
                    M128 = tform.params[0:2, :]  
                    IM128 = cv2.invertAffineTransform(M128)
                    
                    orig_bbox = cv2.transform(np.array([[[0,0], [0,128], [128,0], [128,128]]]), np.array(IM128))                       
                
                    left = floor(min(orig_bbox[0][0][0], orig_bbox[0][1][0] ))
                    if left<0:
                        left=0
                    top = floor(min(orig_bbox[0][0][1], orig_bbox[0][2][1] ))
                    if top<0: 
                        top=0
                    right = ceil(max(orig_bbox[0][2][0], orig_bbox[0][3][0] ))
                    if right>self.video_image.shape[1]:
                        right=self.video_image.shape[1]        
                    bottom = ceil(max(orig_bbox[0][1][1], orig_bbox[0][3][1] ))
                    if bottom>self.video_image.shape[0]:
                        bottom=self.video_image.shape[0]

                    y_diff = bottom-top
                    x_diff = right-left
                
                    crop = self.video_image[int(top):int(bottom),int(left):int(right)]#y,x
                    
                    if y_diff > x_diff:
                        padding = int((y_diff - x_diff) / 2)
                        crop = cv2.copyMakeBorder( crop, 0, 0, padding, padding, cv2.BORDER_CONSTANT)
                    else:
                        padding = int((x_diff - y_diff) / 2)
                        crop = cv2.copyMakeBorder( crop, padding, padding, 0, 0, cv2.BORDER_CONSTANT )
                    crop = cv2.resize( crop, (82, 82))
                    

                    
                    found = False
                    # Test for existing simularities
                    for j in range(len(self.target_faces)):
                        sim = self.findCosineDistance(ret[i].embedding, self.target_faces[j]["Embedding"])
                        
                        if sim<self.parameters["ThresholdAmount"][0]/100.0:
                            found = True
                            
                            self.target_faces[j]["Embedding"] = self.target_faces[j]["Embedding"]*self.target_faces[j]["EmbeddingNumber"] + ret[i].embedding
                            
                            self.target_faces[j]["EmbeddingNumber"] += 1
                            self.target_faces[j]["Embedding"] /=  self.target_faces[j]["EmbeddingNumber"]
                        
                    
                    # If we dont find any existing simularities, it means that this is a new face and should be added to our found faces
                    if not found:
                        new_target_face = self.target_face.copy()
                        self.target_faces.append(new_target_face)
                        last_index = len(self.target_faces)-1

                        self.target_faces[last_index]["TKButton"] = tk.Button(self.found_faces_canvas, self.inactive_button_style, height = 86, width = 86)
                        self.target_faces[last_index]["TKButton"].bind("<MouseWheel>", self.target_faces_mouse_wheel)
                        self.target_faces[last_index]["ButtonState"] = False           
                        self.target_faces[last_index]["Image"] = ImageTk.PhotoImage(image=Image.fromarray(crop))
                        self.target_faces[last_index]["Embedding"] = ret[i].embedding
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

  
    def toggle_source_faces_buttons_state_shift(self, event, button):  
        
        # Set all Source Face buttons to False 
        for face in self.source_faces:      
            face["TKButton"].config(self.inactive_button_style)

        # Toggle the selected Source Face
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

            # else:
                # self.source_faces[button]["TKButton"].config(self.inactive_button_style)

        self.add_action("target_faces", self.target_faces, True, False)
    
    def populate_target_videos(self):
        directory =  self.json_dict["source videos"]

        filenames = os.listdir(directory)        
        
        videos = []
        images = []
        self.target_media = []
        self.target_media_buttons = []
        self.target_media_canvas.delete("all")  
        
        for name in filenames: 
            media_file = os.path.join(directory, name)
            media_object = cv2.imread(media_file)
            
            if media_object is not None:
                image = cv2.cvtColor(media_object, cv2.COLOR_BGR2RGB)            
                image = cv2.resize(image, (82, 82))
                temp = [image, media_file]
                images.append(temp)   

            elif media_object is None:
                media_object = cv2.VideoCapture(media_file)
                
                if media_object.isOpened():                   
                    media_object.set(cv2.CAP_PROP_POS_FRAMES, int(media_object.get(cv2.CAP_PROP_FRAME_COUNT)/2))
                    success, video_frame = media_object.read()
                    
                    if success:
                        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)   
                        
                        
                        im_ratio = float(video_frame.shape[0]) / video_frame.shape[1]

                        if im_ratio>1:
                            new_height = 82
                            new_width = int(new_height / im_ratio)
                        else:
                            new_width = 82
                            new_height = int(new_width * im_ratio)
                        det_scale = float(new_height) / video_frame.shape[0]
                        video_frame = cv2.resize(video_frame, (new_width, new_height))
                        
                        det_img = np.zeros( (82, 82, 3), dtype=np.uint8 )
                        video_frame[:new_height, :new_width, :] = video_frame

                        temp = [video_frame, media_file]
                        videos.append(temp)
                        media_object.release()
                
                else:
                    print('Bad file:', media_file)


        
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

    def load_target(self, button, media_file, media_type):
        self.video_loaded = True

        if media_type == 'Videos':
            self.video_slider.set(0)
            self.add_action("load_target_video", media_file, True)

        elif media_type == 'Images':
            self.add_action("load_target_image", media_file, True)
            self.image_file_name = os.path.splitext(os.path.basename(media_file))
    
        self.set_status(media_file) 
        for i in range(len(self.target_media_buttons)):
            self.target_media_buttons[i].config(self.inactive_button_style)
        
        self.target_media_buttons[button].config(self.button_highlight_style)
        
        if self.actions['SwapFacesState'] == True:
            self.toggle_swapper()
        
        if self.play_video == True:
            self.toggle_play_video()
        
        self.clear_faces()
        
        # delete all markers
        for i in range(len(self.markers)):
            self.video_slider_canvas.delete(self.markers[i]['icon_ref'])
        
        self.markers = []
        self.add_action("markers", self.markers)
      
            
    # @profile
    def set_image(self, image, requested):
        self.video_image = image[0]
        frame = image[1]
        if not requested:
            self.set_slider_position(frame)
            self.parameter_update_from_marker(frame)

        image = self.video_image

        x1 = float(self.x1)        
        y1 = float(self.y1 )
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
        image = ImageTk.PhotoImage(image)
        self.video.configure(image=image)
        self.video.image = image
    
    # @profile    
    def resize_image(self):
    
        image = self.video_image

        x1 = float(self.x1)        
        y1 = float(self.y1 )
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
        image = ImageTk.PhotoImage(image)
        self.video.configure(image=image)
        self.video.image = image
        
    # @profile
    def check_for_video_resize(self):
        if self.video_loaded:
            if self.x1 != self.video.winfo_width() or self.y1 != self.video.winfo_height():
                self.x1 = self.video.winfo_width()
                self.y1 = self.video.winfo_height()
                
                # redisplay markers
                width = self.video_slider_canvas.winfo_width()-30
                for i in range(len(self.markers)):
                    position = 15+int(width*self.markers[i]['frame']/self.video_slider.configure('to')[4])
                    self.video_slider_canvas.delete(self.markers[i]['icon_ref'])
                    self.markers[i]['icon_ref'] = self.video_slider_canvas.create_image(position, 30, image=self.marker_icon)
                
                
                
                if np.any(self.video_image):
                    self.resize_image()
   
    def get_action(self):
        action = self.action_q[0]
        self.action_q.pop(0)
        return action
        
    def get_action_length(self):
        return len(self.action_q)
      

        
    def set_video_slider_length(self, video_length):
        self.video_slider.configure(to=video_length)

    def set_slider_position(self, position):
        self.video_slider.set(position)
   
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



    def toggle_play_video(self):
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
    
    
    def toggle_swapper(self):
        self.actions['SwapFacesState'] = not self.actions['SwapFacesState']
        
        if not self.actions['SwapFacesState']:
            self.actions['SwapFacesButton'].config(self.inactive_button_style)
        else:
            self.actions['SwapFacesButton'].config(self.active_button_style) 

        self.add_action("swap", self.actions['SwapFacesState'], True)
        # if self.actions['ImgVidMode'] == 0:
            # self.add_action('get_requested_video_frame', self.video_slider.get())
        # elif self.actions['ImgVidMode'] == 1:
            # self.add_action('get_requested_video_frame_parameters', None)
            
    def toggle_rec_video(self):
        if not self.play_video:
            self.rec_video = not self.rec_video
                
            if self.rec_video == False:
                self.actions['RecordButton'].config(self.inactive_button_style)
            else:
                self.actions['RecordButton'].config(self.active_button_style, bg='red') 
                
        
            
            

        
    # def set_faceapp_model(self, faceapp):
        # self.faceapp_model = faceapp
 
    def update_CLIP_text(self):
        self.parameters['CLIPText'] = self.temptkstr.get()
        self.add_action("parameters", self.parameters, True)
    
    def add_action(self, action, parameter=None, request_updated_frame=False, ignore_markers=True):
        
        self.action_q.append([action, parameter]) 
        
        if not self.play_video and request_updated_frame:

            if self.actions['ImgVidMode'] == 0 and not ignore_markers:
                self.action_q.append(["get_requested_video_frame", self.video_slider.get()])
            else:
                self.action_q.append(["get_requested_video_frame_parameters", self.video_slider.get()])

    def toggle_dock(self):
        self.dock = False
        if not self.dock:
            # self.video_frame.winfo_width()
            self.grid_rowconfigure(0, weight = 0)
            # self.geometry('%dx%d+%d+%d' % (800, 800, self.winfo_screenwidth()/2-400, self.winfo_screenheight()/2-400))            
            self.geometry('%dx%d' % (self.winfo_width(), 458))         
            self.resizable(width=True, height=False) 

            
            self.undock = self.wm_manage(self.video_frame) 

            self.video_frame.config(width=1024, height=768)
            self.video_frame.grid_propagate(0)
    def set_status(self, msg):
        self.status_label.configure(text=str(msg))
        self.status_label.pack()
        
    def mouse_wheel(self, event, frame):    
        if event.delta > 0: 
            frame += 1
        else:
            frame -= 1

        self.video_slider.set(frame)
        self.add_action("get_requested_video_frame", frame)
        
        self.parameter_update_from_marker(frame)
            
    def save_selected_source_faces(self, text):

        temp = 0
        temp_len = 1
        temp_data = False
        if text != "":
            for i in range(len(self.source_faces)):
                if self.source_faces[i]["ButtonState"]:
                    temp_data = True
                    if temp == []:
                        temp = self.source_faces[i]["Embedding"]
                    else:
                        temp += self.source_faces[i]["Embedding"]
                        temp_len += 1
            
            temp /= temp_len
            
            if temp_data:
                with open("merged_embeddings.txt", "a") as embedfile:
                    identifier = "Name: "+text.get()
                    embedfile.write("%s\n" % identifier)
                    for number in temp:
                        embedfile.write("%s\n" % number)

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

    # def toggle_parameter(self, parameter):
        # self.parameters[parameter] = not self.parameters[parameter]
        
        # if self.parameters[parameter]:
            # self.parameters_buttons[parameter].config(self.active_button_style)
        # else:
            # self.parameters_buttons[parameter].config(self.inactive_button_style)
            
        # self.add_action_and_update_frame("parameters", self.parameters)
        
    def toggle_ui_button(self, parameter):
        state = parameter+'State'
        button = parameter+'Button'
        
        self.parameters[state] = not self.parameters[state]
        
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
        self.add_action('load_null')
        
        # Reset relavent GUI
        if self.actions['SwapFacesState'] == True:
            self.toggle_swapper()
        
        if self.play_video == True:
            self.toggle_play_video()
        
        self.clear_faces()
        
        # delete all markers
        for i in range(len(self.markers)):
            self.video_slider_canvas.delete(self.markers[i]['icon_ref'])
        
        self.markers = []
        self.add_action("markers", self.markers)        
        
        
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
        
        elif parameter == 'SaveImage':   
            resized_image= img.resize((30,30), Image.ANTIALIAS)
            self.actions[icon_holder] = ImageTk.PhotoImage(resized_image)
            self.actions[button] = tk.Button( root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.save_image())            
       
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
            
        elif parameter == 'StartRope':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.load_all())
        elif parameter == 'OutputFolder':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w', command=lambda: self.select_save_video_path())
        elif parameter == 'Threads':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w')
            self.actions[button].bind("<MouseWheel>", self.change_threads_amount)  
        elif parameter == 'VideoQuality':
            self.actions[button] = tk.Button(root, self.inactive_button_style, compound='left', image=self.actions[icon_holder], anchor='w')
            self.actions[button].bind("<MouseWheel>", self.change_video_quality) 
            
        temp = ' '+mode

        self.actions[button].config(text=temp) 
        self.actions[button].place(x=x, y=y, width=width, height=height)        

        # Status Text
        self.actions[button].bind('<Enter>', lambda event: self.set_status(self.actions[message]))
        self.actions[button].bind('<Leave>', lambda event: self.set_status(''))
        
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
            self.update_ui_button('Mask')
            self.update_ui_button('MaskBlur')
            self.update_ui_button('CLIP')
            self.update_ui_button('Occluder')
            self.update_ui_button('FaceParser')
            self.update_ui_button('Blur')
            self.update_ui_button('Threshold')
            self.update_ui_button('Strength')
            self.update_ui_button('Orientation')
            
            self.CLIP_text.delete(0, tk.END)
            self.CLIP_text.insert(0, self.parameters['CLIPText'])

            

    

        
        