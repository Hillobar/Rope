import os
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import json



class GUI(tk.Tk):
    def __init__( self ):  
        super().__init__()
        # Adding a title to the self
        # self.call('tk', 'scaling', 0.5)
        self.title("Test Application")

        self.source_faces = []
        self.found_faces = []
        self.target_videos = []
        self.target_video_file = []
        self.action_q = []
        self.video_image = []
        self.x1 = []
        self.y1 = []
        self.found_faces_assignments = []
        self.play_video = False
        self.rec_video = False
        self.swap = False
        self.faceapp_model = []
        self.GFPGAN_int = tk.IntVar()
        self.fake_diff_int = tk.IntVar()
        self.CLIP_int = tk.IntVar()
        self.video_loaded = False

        self.save_file = []
        self.json_dict = {"source videos":None, "source faces":None, "saved videos":None}
        
        # Video frame
        self.video_frame = tk.Frame( self, bg='grey20')
        self.video_frame.grid( row = 0, column = 0, sticky='NEWS', pady = 2 )
        
        self.video_frame.grid_columnconfigure(0, minsize = 10)  
        self.video_frame.grid_columnconfigure(1, weight = 10) 
        self.video_frame.grid_rowconfigure(0, weight = 1)
        self.video_frame.grid_rowconfigure(1, weight = 0)  
        
        # Video [0,0]
        self.video = tk.Label( self.video_frame, bg='grey50')
        self.video.grid( row = 0, column = 0, columnspan = 3, sticky='NEWS', pady = 2 )
        
        # Video button canvas
        self.video_button_canvas = tk.Canvas( self.video_frame, width = 110, height = 10, bg='grey50' )
        self.video_button_canvas.grid( row = 1, column = 0, sticky='NEWS', pady = 5)
        
        # Video Swap
        self.video_swap = tk.Button( self.video_button_canvas, bg='grey50', text="SWP", command=lambda: self.toggle_swapper())
        self.video_swap.place(x=5, y=5, width = 33, height = 33)   
        
        # Video Play
        self.video_play = tk.Button( self.video_button_canvas, bg='grey50', text="PLY", command=lambda: self.toggle_play_video())
        self.video_play.place(x=40, y=5, width = 33, height = 33)    
     
        # Video Record
        self.video_record = tk.Button( self.video_button_canvas, bg='grey50', text="REC", command=lambda: self.toggle_rec_video())
        self.video_record.place(x=75, y=5, width = 33, height = 33)   
               
        # Video Slider
        self.video_slider = tk.Scale( self.video_frame, orient='horizontal', bg='grey50')
        self.video_slider.bind("<B1-Motion>", lambda event:self.add_action("set_video_position", self.video_slider.get()))
        self.video_slider.bind("<ButtonPress-1>", lambda event:self.add_action("set_video_position", self.video_slider.get()))
        self.video_slider.bind("<ButtonRelease-1>", lambda event:self.add_action("set_video_position", self.video_slider.get()))
        self.video_slider.grid( row = 1, column = 1, sticky='NEWS', pady = 5 )

 ######## Found Faces           
        # Found Faces frame [1,0]
        self.found_faces_frame = tk.Frame( self, bg='grey20')
        self.found_faces_frame.grid( row = 1, column = 0, sticky='NEWS', pady = 2 )
        
        self.found_faces_frame.grid_columnconfigure( 0, minsize = 10 ) 
        self.found_faces_frame.grid_columnconfigure( 1, weight = 1 ) 
        
        self.found_faces_frame.grid_rowconfigure( 0, weight = 0 )  
        self.found_faces_frame.grid_rowconfigure( 1, minsize = 21 )  
        
        # Button Canvas [0,0]
        self.found_faces_buttons = []
        self.found_faces_buttons_canvas = tk.Canvas( self.found_faces_frame, height = 105, width = 125, bg='grey50' )
        self.found_faces_buttons_canvas.grid( row = 0, column = 0, sticky='NEWS')
        
        # Faces Load
        self.found_faces_load_button = tk.Button(self.found_faces_buttons_canvas, bg='grey75', text="Find faces", command=lambda: self.add_action("find_faces", "current"))
        self.found_faces_load_button.place(x=5, y=5, width = 120, height = 20)        
        
        # Faces Clear
        self.found_faces_load_button = tk.Button(self.found_faces_buttons_canvas, bg='grey75', text="Clear", command=lambda: self.add_action("clear_faces", "current"))
        self.found_faces_load_button.place(x=5, y=30, width = 120, height = 20)    
        
        # Faces Canvas [0,1]
        self.found_faces_canvas = tk.Canvas( self.found_faces_frame, height = 105, bg='grey50' )
        self.found_faces_canvas.grid( row = 0, column = 1, sticky='NEWS')
        
        # Faces Scrollbar [0..1, 1]
        self.found_faces_scrollbar = tk.Scrollbar( self.found_faces_frame, orient="horizontal", command = self.found_faces_canvas.xview )
        self.found_faces_scrollbar.grid( row = 1, column = 0, columnspan = 2, sticky='NEWS')
        self.found_faces_canvas.configure( xscrollcommand = self.found_faces_scrollbar.set )        
                
 ######## Source Faces       
        # Source Faces frame [2,0]
        self.source_faces_frame = tk.Frame( self, bg='grey20')
        self.source_faces_frame.grid( row = 2, column = 0, sticky='NEWS', pady = 2 )

        self.source_faces_frame.grid_columnconfigure( 0, minsize = 10 ) 
        self.source_faces_frame.grid_columnconfigure( 1, weight = 1 ) 
        
        self.source_faces_frame.grid_rowconfigure( 0, weight = 0 )  
        self.source_faces_frame.grid_rowconfigure( 1, minsize = 21 )  
        
        # Button Canvas [0,0]
        self.source_faces_buttons = []
        self.source_button_canvas = tk.Canvas( self.source_faces_frame, height = 105, width = 125, bg='grey50' )
        self.source_button_canvas.grid( row = 0, column = 0, sticky='NEWS')
        
        # Faces Load
        self.source_faces_load_button = tk.Button(self.source_button_canvas, bg='grey75', text="Reload faces", command=lambda: self.populate_faces_canvas())
        self.source_faces_load_button.place(x=5, y=5, width = 120, height = 20) 
        
        # Source Faces Filepath
        self.faces_filepath_button = tk.Button(self.source_button_canvas, bg='gold', text="Set faces folder", wraplength=115, command=lambda: self.select_faces_path())
        self.faces_filepath_button.place(x=5, y=30, width = 120, height = 75) 

        # Faces Canvas [0,1]
        self.source_faces_canvas = tk.Canvas( self.source_faces_frame, height = 105, bg='grey50' )
        self.source_faces_canvas.grid( row = 0, column = 1, sticky='NEWS')
        
        # Faces Scrollbar [0..1, 1]
        self.source_faces_scrollbar = tk.Scrollbar( self.source_faces_frame, orient="horizontal", command = self.source_faces_canvas.xview )
        self.source_faces_scrollbar.grid( row = 1, column = 0, columnspan = 2, sticky='NEWS')
        self.source_faces_canvas.configure( xscrollcommand = self.source_faces_scrollbar.set )

######### Target Videos
        # Target Video frame [3,0]
        self.target_videos_frame = tk.Frame( self, bg='grey20')
        self.target_videos_frame.grid( row = 3, column = 0, sticky='NEWS', pady = 2 )
        
        self.target_videos_frame.grid_columnconfigure( 0, minsize = 10 ) 
        self.target_videos_frame.grid_columnconfigure( 1, weight = 1 ) 
        
        self.target_videos_frame.grid_rowconfigure( 0, weight = 0 )  
        self.target_videos_frame.grid_rowconfigure( 1, minsize = 21 )  
        
        # Button Canvas [0,0]
        self.target_videos_buttons = []
        self.target_button_canvas = tk.Canvas( self.target_videos_frame, height = 105, width = 125, bg='grey50' )
        self.target_button_canvas.grid( row = 0, column = 0, sticky='NEWS')
        
        # Videos Load
        self.target_video_load_button = tk.Button(self.target_button_canvas, bg='grey75', text="Reload videos", command=lambda: self.populate_target_videos())
        self.target_video_load_button.place(x=5, y=5, width = 120, height = 20)    

        # Source Videos Filepath
        self.video_filepath_button = tk.Button(self.target_button_canvas, bg='gold', text="Set videos folder", wraplength=115, command=lambda: self.select_video_path())
        self.video_filepath_button.place(x=5, y=30, width = 120, height = 75)            
        
        # Video Canvas [0,1]
        self.target_video_canvas = tk.Canvas( self.target_videos_frame, height = 105, bg='grey50' )
        self.target_video_canvas.grid( row = 0, column = 1, sticky='NEWS')
        
        # Video Scrollbar [0..1, 1]
        self.target_video_scrollbar = tk.Scrollbar( self.target_videos_frame, orient="horizontal", command = self.target_video_canvas.xview )
        self.target_video_scrollbar.grid( row = 1, column = 0, columnspan = 2, sticky='NEWS')
        self.target_video_canvas.configure( xscrollcommand = self.target_video_scrollbar.set )

######### Options
        self.options_frame = tk.Frame( self, bg='grey20', height = 128)
        self.options_frame.grid( row = 4, column = 0, sticky='NEWS', pady = 2 )
        
        self.options_frame.grid_rowconfigure( 0, weight = 100 )          
        self.options_frame.grid_columnconfigure( 0, weight = 1 ) 
        
        # Left Canvas
        self.options_frame_canvas1 = tk.Canvas( self.options_frame, height = 105, bg='grey50' )
        self.options_frame_canvas1.grid( row = 0, column = 0, sticky='NEWS', pady = 0 )
        
        # Label Frame 1
        self.label_frame1 = tk.LabelFrame( self.options_frame_canvas1, height = 100, width = 165, bg='grey50' )
        self.label_frame1.place(x=5, y=5)
        
        # GFPGAN
        self.gfpgan_checkbox = tk.Checkbutton(self.label_frame1, anchor="w", text='GFPGAN',variable=self.GFPGAN_int, bg='grey75', height=1, onvalue=True, offvalue=False, command=lambda: self.add_action_and_update_frame("gfpgan_checkbox", self.GFPGAN_int.get()))
        self.gfpgan_checkbox.place(x=5, y=8, width=80, height=17)

        self.GFPGAN_blend = tk.Spinbox( self.label_frame1, from_=0, to=100, increment = 5, width = 5 ,bd = 4, command=lambda :self.add_action_and_update_frame("GFPGAN_blend",self.GFPGAN_blend.get()  ))
        self.GFPGAN_blend.place(x=100, y=5)
        self.GFPGAN_blend.insert(0,5)
        
        # Fake_diff
        self.fake_diff_checkbox = tk.Checkbutton(self.label_frame1, anchor="w", text='Diffing',variable=self.fake_diff_int, bg='grey75', onvalue=True, offvalue=False, command=lambda: self.add_action_and_update_frame("fake_diff_checkbox", self.fake_diff_int.get()))
        self.fake_diff_checkbox.place(x=5, y=38, width=80, height=17)
        
        self.fake_diff_blend = tk.Spinbox( self.label_frame1, from_=0, to=255, increment = 1, width = 5 ,bd = 4, command=lambda :self.add_action_and_update_frame("fake_diff_blend",self.fake_diff_blend.get()  ))
        self.fake_diff_blend.place(x=100, y=35)
        
        # Face Threshhold
        self.face_thresh_id = tk.Label(self.label_frame1, anchor="w", bg='grey75', text="Threshhold")
        self.face_thresh_id.place(x=5, y=68, width=80, height=17)
        
        var = tk.DoubleVar(value=0.85)
        self.face_thresh = tk.Spinbox( self.label_frame1, from_=0, to=1, increment = 0.01, width = 5 ,bd = 4, textvariable=var, command=lambda :self.add_action_and_update_frame("face_thresh",float(self.face_thresh.get())))
        self.face_thresh.place(x=100, y=65)
         
        # Label Frame 2
        self.label_frame2 = tk.LabelFrame( self.options_frame_canvas1, height = 100, width = 130, bg='grey50' )
        self.label_frame2.place(x=170, y=5)
        
        self.top_blend_id = tk.Label(self.label_frame2, anchor="w", bg='grey75', text="Mask top:")
        self.top_blend_id.place(x=5, y=8, width=60, height=17)
        
        var = tk.IntVar(value=10)
        self.top_blend_scrollbar = tk.Spinbox( self.label_frame2, from_=0, to=255, increment = 1, width = 5 ,bd = 4, textvariable=var, command=lambda : self.add_action_and_update_frame("top_blend", self.top_blend_scrollbar.get()) )
        self.top_blend_scrollbar.place(x=70, y=5)
        
        # var = tk.IntVar(value=10)
        # self.bottom_blend_scrollbar = tk.Spinbox( self.label_frame2, from_=0, to=255, increment = 1, width = 5 ,bd = 4, textvariable=var,  command=lambda : self.add_action_and_update_frame("bottom_blend",self.bottom_blend_scrollbar.get() ))
        # self.bottom_blend_scrollbar.place(x=30, y=65)
        
        self.left_blend_id = tk.Label(self.label_frame2, anchor="w", bg='grey75', text="Mask side:")
        self.left_blend_id.place(x=5, y=38, width=60, height=17)
        
        var = tk.IntVar(value=10)
        self.left_blend_scrollbar = tk.Spinbox( self.label_frame2, from_=0, to=255, increment = 1, width = 5 ,bd = 4, textvariable=var,  command=lambda : self.add_action_and_update_frame("left_blend",self.left_blend_scrollbar.get() ))
        self.left_blend_scrollbar.place(x=70, y=35)
        
        # var = tk.IntVar(value=8)
        # self.right_blend_scrollbar = tk.Spinbox( self.label_frame2, from_=0, to=255, increment = 1, width = 5 ,bd = 4, textvariable=var,  command=lambda : self.add_action_and_update_frame("right_blend",self.right_blend_scrollbar.get() ))
        # self.right_blend_scrollbar.place(x=60, y=35)
        
        self.mask_blur_id = tk.Label(self.label_frame2, anchor="w", bg='grey75', text="Mask blur:")
        self.mask_blur_id.place(x=5, y=68, width=60, height=17)
        
        var = tk.IntVar(value=10)
        self.mask_blur_scrollbar = tk.Spinbox( self.label_frame2, from_=0, to=128, increment = 1, width = 5 ,bd = 4, textvariable=var,  command=lambda : self.add_action_and_update_frame("blur",self.mask_blur_scrollbar.get() ))
        self.mask_blur_scrollbar.place(x=70, y=65)
        
        # var = tk.IntVar(value=20)
        # self.blur_scrollbar = tk.Scale( self.label_frame2, orient="vertical", showvalue=0, length = 80, command=lambda i: self.add_action_and_update_frame("blur",int(i)) )
        # self.blur_scrollbar.place(x=120, y=5)
        # self.blur_scrollbar.set(20)
        
        # Label Frame 3
        self.label_frame3 = tk.LabelFrame( self.options_frame_canvas1, height = 100, width = 260, bg='grey50' )
        self.label_frame3.place(x=300, y=5)
        
        # CLIPs
        self.CLIP_checkbox = tk.Checkbutton(self.label_frame3, anchor="w", text='CLIP',variable=self.CLIP_int, bg='grey75', onvalue=True, offvalue=False, command=lambda: self.add_action_and_update_frame("CLIP_checkbox", self.CLIP_int.get()))
        self.CLIP_checkbox.place(x=5, y=8, height=17 )        
        
        self.create_clip_button = tk.Button(self.label_frame3, bg='grey75', text="Apply", command=lambda: self.apply_CLIPs())
        self.create_clip_button.place(x=65, y=5, width = 83, height = 25)     
  
        # Blur
        self.CLIP_blur_label = tk.Label(self.label_frame3, anchor="w", bg='grey75', text="Blur:")
        self.CLIP_blur_label.place(x=155, y=8, width = 35, height=17 )

        var = tk.DoubleVar(value=8)
        self.CLIP_blur = tk.Spinbox( self.label_frame3, from_=0, to=128, increment = 1, width = 5 ,bd = 4, textvariable=var, command=lambda :self.add_action_and_update_frame("CLIP_blur",float(self.CLIP_blur.get())))
        self.CLIP_blur.place(x=195, y=5) 
       
        # Pos
        self.CLIP_pos_text_label = tk.Label(self.label_frame3, anchor="w", height=1, bg='grey75', text="Pos:")
        self.CLIP_pos_text_label.place(x=5, y=38, width = 30, height=17) 
        
        self.CLIP_pos_text = tk.Text(self.label_frame3, height=1)
        self.CLIP_pos_text.place(x=40, y=38, width = 150, height=17) 
        
        var = tk.DoubleVar(value=0.5)
        self.pos_thresh = tk.Spinbox( self.label_frame3, from_=0, to=1, increment = 0.01, width = 5 ,bd = 4, textvariable=var, command=lambda :self.add_action_and_update_frame("pos_thresh",float(self.pos_thresh.get())))
        self.pos_thresh.place(x=195, y=35)
        
        # Neg
        self.CLIP_neg_text_label = tk.Label(self.label_frame3, anchor="w", bg='grey75', text="Neg:")
        self.CLIP_neg_text_label.place(x=5, y=68, width = 30, height=17)
       
        self.CLIP_neg_text = tk.Text(self.label_frame3)
        self.CLIP_neg_text.place(x=40, y=68, width = 150, height=17) 

        var = tk.DoubleVar(value=0.5)
        self.neg_thresh = tk.Spinbox( self.label_frame3, from_=0, to=1, increment = 0.01, width = 5 ,bd = 4, textvariable=var, command=lambda :self.add_action_and_update_frame("neg_thresh",float(self.neg_thresh.get())))
        self.neg_thresh.place(x=195, y=65)
        



       
        # Label Frame 4
        self.label_frame4 = tk.LabelFrame( self.options_frame_canvas1, height = 100, width = 210, bg='grey50' )
        self.label_frame4.place(x=585, y=5)
        
        # Load all
        self.create_video_button = tk.Button(self.label_frame4, bg='gold', text="Load Models", command=lambda: self.load_all())
        self.create_video_button.place(x=5, y=5, width = 195, height = 25) 
 
        # Save Videos Filepath
        self.save_video_filepath_button = tk.Button(self.label_frame4, bg='gold', text="Set save videos folder", wraplength=115, command=lambda: self.select_save_video_path())
        self.save_video_filepath_button.place(x=5, y=35, width = 195, height = 25)    

        # Threads
        var = tk.IntVar(value=5)
        self.num_threads = tk.Spinbox( self.label_frame4, from_=0, to=20, increment = 1, width = 5 ,bd = 4, textvariable=var, command=lambda :self.add_action("num_threads",int(self.num_threads.get())))
        self.num_threads.place(x=75, y=65)
        
        self.num_threads_id = tk.Label(self.label_frame4, anchor="w", bg='grey75', text="# Threads:")
        self.num_threads_id.place(x=5, y=68, width = 65, height=17)

        
       

    def initialize_gui( self ):
        self.geometry("800x800")
        self.title("Rope")
        self.configure(bg='grey10')
        self.resizable(width=True, height=True)        

        self.grid_columnconfigure(0, weight = 1)  

        self.grid_rowconfigure(0, weight = 10)
        self.grid_rowconfigure(1, weight = 0)  
        self.grid_rowconfigure(2, weight = 0)  
        self.grid_rowconfigure(3, weight = 0)
        self.grid_rowconfigure(4, weight = 0)     

        self.add_action("num_threads", int(self.num_threads.get()))
        self.add_action("face_thresh", float(self.face_thresh.get()))
        self.add_action("top_blend", self.top_blend_scrollbar.get())
        self.add_action("bottom_blend",10 )
        self.add_action("left_blend",self.left_blend_scrollbar.get() )
        self.add_action("right_blend",self.left_blend_scrollbar.get() )
        # self.add_action("blur",self.blur_scrollbar.get())
        self.add_action_and_update_frame("blur",self.mask_blur_scrollbar.get() )
        self.add_action_and_update_frame("CLIP_blur",float(self.CLIP_blur.get()))
        
        self.add_action("CLIP_checkbox", self.CLIP_int.get())
        self.add_action("GFPGAN_blend",self.GFPGAN_blend.get())
        
        self.add_action("pos_thresh",float(self.neg_thresh.get()))
        self.add_action("neg_thresh",float(self.neg_thresh.get()))

        try:
            self.save_file = open("data.json", "r")
        except:
            with open("data.json", "w") as outfile:
                json.dump(self.json_dict, outfile)
        else:
            jason_object = []
            with open('data.json', 'r') as openfile:
                json_object = json.load(openfile)
            
            self.json_dict["source videos"] = json_object["source videos"]
            if self.json_dict["source videos"]:
                self.video_filepath_button.configure(bg='grey75', text=self.json_dict["source videos"])
        
            self.json_dict["source faces"] = json_object["source faces"]
            if self.json_dict["source faces"]:
                self.faces_filepath_button.configure(bg='grey75', text=self.json_dict["source faces"])
            
            self.json_dict["saved videos"] = json_object["saved videos"]
            if self.json_dict["saved videos"]:
                self.save_video_filepath_button.configure(bg='grey75', text=self.json_dict["saved videos"])
                self.add_action("saved_video_path",self.json_dict["saved videos"])
                
            
            
    def load_all(self):
        if not self.json_dict["source videos"] or not self.json_dict["source faces"]:
            print("Please set faces and videos folders first!")
            return
        
        self.add_action("load_models", True)
        self.create_video_button.configure( bg='grey75', text="Models loaded!")
        
        
    def select_video_path(self):
         
        self.json_dict["source videos"] = filedialog.askdirectory(title="Select a target")
         
        self.video_filepath_button.configure(bg='grey75', text=self.json_dict["source videos"]) 
        
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)
            
            
    def select_save_video_path(self):
        self.json_dict["saved videos"] = filedialog.askdirectory(title="Select a target")
         
        self.save_video_filepath_button.configure(bg='grey75', text=self.json_dict["saved videos"]) 
        
        self.add_action("saved_video_path",self.json_dict["saved videos"])
        
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)    

    def select_faces_path(self):

        self.json_dict["source faces"] = filedialog.askdirectory(title="Select a target")
        
        self.faces_filepath_button.configure(bg='grey75', text=self.json_dict["source faces"])
         
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)
            
    def populate_faces_canvas(self):
        if not self.faceapp_model:
            print("Load model first")
        else:
            self.source_faces_buttons = []
            self.source_faces = []    
            self.source_faces_canvas.delete("all")

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
                        bbox = ret[0].bbox
                        y_diff = bbox[3] - bbox[1]
                        x_diff = bbox[2] - bbox[0]
                    
                        crop = temp_file[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]#y,x
                        if y_diff > x_diff:
                            padding = int((y_diff - x_diff) / 2)
                            crop = cv2.copyMakeBorder( crop, 0, 0, padding, padding, cv2.BORDER_CONSTANT)
                        else:
                            padding = int((x_diff - y_diff) / 2)
                            crop = cv2.copyMakeBorder( crop, padding, padding, 0, 0, cv2.BORDER_CONSTANT )
                                    
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)            
                        crop = cv2.resize( crop, (85, 85))
                        temp = [crop, ret[0].embedding]
                        faces.append(temp)
                                                   
                # Add faces[] images to buttons
                for i in range(len(faces)):              
                    rgb_face = Image.fromarray(faces[i][0])
                    temp = [ImageTk.PhotoImage(image=rgb_face), faces[i][1]]
                    self.source_faces.append(temp)  
                    temp = [tk.Button(self.source_faces_canvas, image = self.source_faces[i][0], height = 100, width = 100, anchor=tk.N, bg='grey75'), False]
                    self.source_faces_buttons.append(temp)  
                    self.source_faces_buttons[i][0].config(text = str(i), command=lambda i=i: self.toggle_source_faces_buttons_state(i))          
                    self.source_faces_canvas.create_window(i*105,0, window = self.source_faces_buttons[i][0],anchor=tk.SW)
                
                self.source_faces_canvas.configure(scrollregion = self.source_faces_canvas.bbox("all"))
                
                # send over source faces embeddings
                # self.source_faces[i][0=photoimage, 1=embedding]
                self.add_action("source_embeddings", self.source_faces)
        
    def find_faces(self, scope):
        try:
            ret = self.faceapp_model.get(self.video_image, max_num=10)
        except Exception:
            print(" No video selected")
        else:    
           
            faces = []  
            
            # Find all faces and add to faces[]
            if ret:
                # Loop thgouh all faces in video frame
                for i in range(len(ret)):
                    # Create a frame for each face
                    bbox = ret[i].bbox

                    if bbox[0] < 0:
                        bbox[0] = 0
                    if bbox[1] < 0:
                        bbox[1] = 0
                    if bbox[2]>self.video_image.shape[1]:
                        bbox[2] = self.video_image.shape[1]
                    if bbox[3]>self.video_image.shape[0]:
                        bbox[3] = self.video_image.shape[0]
                            
                            
                    y_diff = bbox[3] - bbox[1]
                    x_diff = bbox[2] - bbox[0]

                    crop = self.video_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]#y,x
                    
                    if y_diff > x_diff:
                        padding = int((y_diff - x_diff) / 2)
                        crop = cv2.copyMakeBorder( crop, 0, 0, padding, padding, cv2.BORDER_CONSTANT)
                    else:
                        padding = int((x_diff - y_diff) / 2)
                        crop = cv2.copyMakeBorder( crop, padding, padding, 0, 0, cv2.BORDER_CONSTANT )
                    crop = cv2.resize( crop, (85, 85))
                    
                    found = False
                    # Test for existing simularities
                    for j in range(len(self.found_faces)):
                        sim = self.findCosineDistance(ret[i].embedding, self.found_faces[j][1])
                        if sim<0.85:
                            found = True
                    # If we dont find any existing simularities, it means that this is a new face and should be added to our found faces
                    if not found:
                        temp = [crop, ret[i].embedding]
                        faces.append(temp)
                        temp = [ret[i].embedding, -1]
                        self.found_faces_assignments.append(temp)

            prev_len = len(self.found_faces)    
     
            # Add faces[] images to buttons
            # append to found_faces[] holder
            # add buttons
            for i in range(len(faces)):                 
                rgb_face = Image.fromarray(faces[i][0])
                temp = [ImageTk.PhotoImage(image=rgb_face), faces[i][1], -1]
                self.found_faces.append(temp) 
                temp = [tk.Button(self.found_faces_canvas, height = 100, width = 100, anchor=tk.N, bg='grey75'), False]
                self.found_faces_buttons.append(temp)
            
            # for added buttons, assign faces
            # for add buttons, add to canvas
            for i in range(len(faces)):              
                self.found_faces_buttons[prev_len+i][0].config( image = self.found_faces[prev_len+i][0], command=lambda i=i+prev_len: self.toggle_found_faces_buttons_state(i) )
                self.found_faces_canvas.create_window((prev_len+i)*100,0, window = self.found_faces_buttons[prev_len+i][0],anchor=tk.SW)            

            self.found_faces_canvas.configure(scrollregion = self.found_faces_canvas.bbox("all")) 
         
    def clear_faces(self):
        self.found_faces_buttons = []
        self.found_faces = []    
        self.found_faces_assignments = []
        self.add_action("found_assignments", self.found_faces_assignments)
        self.found_faces_canvas.delete("all")    
            
            
    # toggle the founf faces button and make assignments        
    def toggle_found_faces_buttons_state(self, button):  
        # Reverse State
        self.found_faces_buttons[button][1] = not self.found_faces_buttons[button][1]
        
        # if button is True make assignment to source faces
        if self.found_faces_buttons[button][1] == True:
            # Check if any source_faces is True
            for i in range(len(self.source_faces_buttons)):
                if self.source_faces_buttons[i][1] == True:
                    self.found_faces[button][2] = i
                    self.found_faces_assignments[button][1] = i
                    self.found_faces_buttons[button][0].config(text = str(i), compound = tk.TOP, relief=tk.SUNKEN) 
        else:
            self.found_faces[button][2] = -1
            self.found_faces_assignments[button][1] = -1
            self.found_faces_buttons[button][0].config(compound = tk.NONE, relief=tk.RAISED)
            
        #send over assignments
        if self.play_video == False:
            self.add_action_and_update_frame("found_assignments", self.found_faces_assignments)
        else:
            self.add_action("found_assignments", self.found_faces_assignments)

    def toggle_source_faces_buttons_state(self, button):  
        self.source_faces_buttons[button][1] = not self.source_faces_buttons[button][1]
        if self.source_faces_buttons[button][1] == True:
            for i in range(len(self.source_faces_buttons)):                
                self.source_faces_buttons[i][1] = False
            self.source_faces_buttons[button][1] = True
        else:
            self.source_faces_buttons[button][1] = False
    
        for i in range(len(self.source_faces_buttons)):
            if self.source_faces_buttons[i][1] == False:
                self.source_faces_buttons[i][0].config(compound = tk.NONE, relief=tk.RAISED)
            else:
                self.source_faces_buttons[button][0].config(compound = tk.TOP, relief=tk.SUNKEN) 
    
    def populate_target_videos(self):

        self.target_videos_buttons = []
        self.target_videos = []    
        self.target_video_canvas.delete("all")

        directory =  self.json_dict["source videos"]

        filenames = os.listdir(directory)
        
        
        videos = []
        self.target_videos = []
        self.target_videos_buttons = []
        self.target_video_canvas.delete("all")  
        
        for name in filenames: #should check if is an image
            video_file = os.path.join(directory, name)
            vidcap = cv2.VideoCapture(video_file)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/2))
            success, image = vidcap.read()
            if success:
                crop = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)            
                crop = cv2.resize( crop, (100, 100))
                temp = [crop, video_file]
                videos.append(temp)

        for i in range(len(videos)):
            self.target_videos_buttons.append(tk.Button(self.target_video_canvas, height = 100, width = 100, bg='grey75'))
                                               
                  
        for i in range(len(videos)):  
            rgb_video = Image.fromarray(videos[i][0])        
            self.target_videos.append(ImageTk.PhotoImage(image=rgb_video))            
            self.target_videos_buttons[i].config( image = self.target_videos[i],  command=lambda i=i: self.load_target_video(i, videos[i][1]))
            self.target_video_canvas.create_window(i*100,0, window = self.target_videos_buttons[i],anchor=tk.SW)
            
        self.target_video_canvas.configure(scrollregion = self.target_video_canvas.bbox("all"))

    def load_target_video(self, button, video_file):
        self.video_loaded = True
        self.add_action("load_target_video", video_file)
        for i in range(len(self.target_videos_buttons)):
            self.target_videos_buttons[i].config(relief=tk.RAISED)
        self.target_videos_buttons[button].config(relief=tk.SUNKEN)
        
        if self.swap == True:
            self.toggle_swapper()
        
        self.clear_faces()
            
    
    def set_image(self, image, requested):
        self.video_image = image[0]
        if not requested:
            self.set_slider_position(image[1])

    def display_image_in_video_frame(self):

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
            padding = ((y1/y2)*x1-x2)/2.0
            padding = int(padding)
            image = cv2.copyMakeBorder( image, padding, padding, 0, 0, cv2.BORDER_CONSTANT)            
        else:
            y2=y1
            x2=y2*m2
            image = cv2.resize(image, (int(x2), int(y2)))
            padding = ((x1/x2)*y1-y2)
            padding = int(padding)
            image = cv2.copyMakeBorder( image, 0, 0, padding, padding, cv2.BORDER_CONSTANT) 

        #image = cv2.resize(image, (int(x1), int(y1)))
        image = Image.fromarray(image)  
        image = ImageTk.PhotoImage(image)
        self.video.configure(image=image)
        self.video.image = image

    def check_for_video_resize(self):
        if self.x1 != self.video.winfo_width() or self.y1 != self.video.winfo_height():
            self.x1 = self.video.winfo_width()
            self.y1 = self.video.winfo_height()
            if np.any(self.video_image):
                self.display_image_in_video_frame()
   
    def get_action(self):
        action = self.action_q[0]
        self.action_q.pop(0)
        return action
        
    def get_action_length(self):
        return len(self.action_q)
      
    def add_action(self, action, param):
        temp = [action, param]
        self.action_q.append(temp)
        
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

    # def process(self):
  
        # swap = False
        # # for i in range(len(self.found_faces)):
            # # if self.found_faces[i][2] != -1:
                # # swap = True
        # # if swap:
            # # self.add_action("swap", True)

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
                    self.video_play.config(relief=tk.RAISED)
                else:
                    self.add_action("play_video", "record")
                    self.video_play.config(relief=tk.SUNKEN)
            else:
                self.add_action("play_video", "play")
                self.video_play.config(relief=tk.SUNKEN)
            
        else:
            self.add_action("play_video", "stop")
            self.video_play.config(relief=tk.RAISED)
            if self.rec_video:
                self.toggle_rec_video()
    
    
    
    def toggle_swapper(self):
        self.swap = not self.swap
        
        if not self.swap:
            self.video_swap.config(relief=tk.RAISED)
        else:
            self.video_swap.config(relief=tk.SUNKEN) 
        
        if not self.play_video:
            self.add_action_and_update_frame("swap", self.swap)
        else:
            self.add_action("swap", self.swap)
            
    def toggle_rec_video(self):
        if not self.play_video:
            self.rec_video = not self.rec_video
                
            if self.rec_video == False:
                self.video_record.config(relief=tk.RAISED, bg='grey50')
            else:
                self.video_record.config(relief=tk.SUNKEN, bg='red') 
                
        
            
            

        
    def set_faceapp_model(self, faceapp):
        self.faceapp_model = faceapp
        
    def send_var(self):
        self.add_action("variable", self.options_scrollbar.get())
        self.add_action("set_video_position", self.video_slider.get())
        
    def add_action_and_update_frame(self, action, parameter):
        self.add_action(action, parameter)
        self.add_action("set_video_position", self.video_slider.get())   

    def apply_CLIPs(self):
        temp = [self.CLIP_pos_text.get(1.0, "end-1c"), self.CLIP_neg_text.get(1.0, "end-1c")]
        self.add_action_and_update_frame("apply_CLIPs", temp)

        
        
