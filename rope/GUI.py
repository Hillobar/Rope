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
        self.pixel = []
        self.target_face = {    
                            "TKButton":                 [],
                            "ButtonState":              "off",
                            "Image":                    [],
                            "Embedding":                [],
                            "SourceFaceAssignments":    [],
                            "EmbeddingNumber":          0
                            }
        self.target_faces = []
        
        self.source_face =  {
                            "TKButton":                 [],
                            "ButtonState":              "off",
                            "Image":                    [],
                            "Embedding":                []
                            }   
        self.source_faces = []

        self.parameters =   {
                            "GFPGANState":              False,
                            "GFPGANAmount":             100,
                            "DiffState":                False,
                            "DiffAmount":               4,
                            "ThreshholdState":          False,
                            "Threshhold":               0.85,
                            "MaskTop":                  20,
                            "MaskSide":                 30,
                            "MaskBlur":                 15,
                            "OccluderState":            False,
                            "CLIPState":                False,
                            "CLIPText":                 tk.StringVar(value=""),
                            "CLIPAmount":               0.5,
                            "FaceParserState":          False,
                            "BlurAmount":               5
                            }
                                
                 


        self.num_threads = 1
        self.video_quality = 18
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
        # self.GFPGAN_int = tk.IntVar()
        # self.fake_diff_int = tk.IntVar()
        # self.CLIP_int = tk.IntVar()
        self.video_loaded = False
        # self.occluder_int = tk.IntVar()
        self.dock = True
        self.undock = []

        self.save_file = []
        self.json_dict = {"source videos":None, "source faces":None, "saved videos":None, "threads":1}
        
        self.new_int = tk.IntVar()
        
        self.button1 = "gray25"
        self.button_1_text = "light goldenrod"
        self.button1_active = "gray50"
        
        self.button_highlight_style =    {  
                                'bg':               'light goldenrod', 
                                'fg':               'gray20', 
                                'activebackground': 'gray75', 
                                'activeforeground': 'light goldenrod',
                                'relief':           'flat',
                                'border':           '0',
                                'font':             ("Arial", 9)
                                }   
        self.inactive_button_style =    {  
                                'bg':               'gray20', 
                                'fg':               'white', 
                                'activebackground': 'gray10', 
                                'activeforeground': 'white',
                                'relief':           'flat',
                                'border':           '0',
                                'font':             ("Arial", 9)
                                }   
        self.active_button_style =    {  
                                'bg':               'black', 
                                'fg':               'white', 
                                'activebackground': 'gray10', 
                                'activeforeground': 'white',
                                'relief':           'flat',
                                'border':           '0',
                                'font':             ("Arial", 9)
                                }
        self.need_button_style =    {  
                                'bg':               'gray30', 
                                'fg':               'white', 
                                'relief':           'flat',
                                'border':           '0',                                	
                                'font':             ("Arial italic", 9)
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
        self.button_style2 =    {  
                                'bg':               'light goldenrod', 
                                } 
        self.button_style3 =    {  
                                'bg':               'gray25', 
                                'relief':           'flat',
                                'border':           '1'
                                } 
 
        self.spinbox_style =    { 
                                'width':            '5',
                                'bg':               'gray40', 
                                'fg':               'white', 
                                'relief':           'flat',
                                'width':            '5'
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
                                'font':             ("Arial", 9)
                                }                                    
                                                    
        # Video frame
        self.video_frame = tk.Frame( self, self.frame_style)
        self.video_frame.grid( row = 0, column = 0, sticky='NEWS', pady = 2 )
        
        self.video_frame.grid_columnconfigure(0, minsize = 10)  
        self.video_frame.grid_columnconfigure(1, weight = 10) 
        self.video_frame.grid_rowconfigure(0, weight = 1)
        self.video_frame.grid_rowconfigure(1, weight = 0)  
        
        # Video [0,0]
        self.video = tk.Label( self.video_frame, self.label_style, bg='black')
        self.video.grid( row = 0, column = 0, columnspan = 3, sticky='NEWS', pady =0 )
        self.video.bind("<MouseWheel>", self.iterate_through_merged_embeddings)
        
        # Video button canvas
        self.video_button_canvas = tk.Canvas( self.video_frame, self.canvas_style1, width = 112, height = 40)
        self.video_button_canvas.grid( row = 1, column = 0, sticky='NEWS', pady = 0)
        
        # Dock
        self.video_dock = tk.Button( self.video_button_canvas, self.inactive_button_style, text="^^", wraplength=1, command=lambda: self.toggle_dock())
        self.video_dock.place(x=8, y=2, width = 15, height = 36)   

        # Video Play
        img = Image.open('./rope/media/play.png')
        resized_image= img.resize((30,30), Image.ANTIALIAS)
        self.play_icon = ImageTk.PhotoImage(resized_image)
        self.video_play = tk.Button( self.video_button_canvas, self.inactive_button_style, image=self.play_icon, command=lambda: self.toggle_play_video())
        self.video_play.place(x=31, y=2, width = 36, height = 36)    

        # Video Record
        img = Image.open('./rope/media/rec.png')
        resized_image= img.resize((30,30), Image.ANTIALIAS)
        self.rec_icon = ImageTk.PhotoImage(resized_image)
        
        self.video_record = tk.Button( self.video_button_canvas, self.inactive_button_style, image=self.rec_icon, command=lambda: self.toggle_rec_video())
        self.video_record.place(x=69, y=2, width = 36, height = 36)   
               
        # Video Slider
        self.video_slider = tk.Scale( self.video_frame, self.slider_style, orient='horizontal')
        self.video_slider.bind("<B1-Motion>", lambda event:self.add_action_and_update_frame("set_video_position", self.video_slider.get(), False))
        self.video_slider.bind("<ButtonPress-1>", lambda event:self.add_action_and_update_frame("set_video_position", self.video_slider.get(), False))
        self.video_slider.bind("<ButtonRelease-1>", lambda event:self.add_action_and_update_frame("set_video_position", self.video_slider.get(), False))
        self.video_slider.bind("<MouseWheel>", self.mouse_wheel)
        self.video_slider.grid( row = 1, column = 1, sticky='NEWS', pady = 2 )

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
        self.label_frame1 = tk.LabelFrame( self.options_frame_canvas1, self.frame_style, height = 71, width = 800 )
        self.label_frame1.place(x=0, y=0)
        
        column1=8
        # GFPGAN
        # GFPGAN-checkbox
        img = Image.open('./rope/media/gfpgan_logo.png')
        resized_image= img.resize((45,20), Image.ANTIALIAS)
        self.GFPGAN_icon = ImageTk.PhotoImage(resized_image)
        temp = '              ' + str(int(self.parameters["GFPGANAmount"])) + '%' 
        self.GFPGAN_button = tk.Button(self.label_frame1, self.inactive_button_style, compound='left', image=self.GFPGAN_icon, text=temp, anchor='w', command=lambda: self.toggle_GFPGAN())
        self.GFPGAN_button.place(x=column1, y=8, width = 125, height = 26) 
        self.GFPGAN_button.bind("<MouseWheel>", self.change_GFPGAN_amount)

        # Fake_diff
        # Fake_diff-checkbox
        img = Image.open('./rope/media/diff.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.diff_icon = ImageTk.PhotoImage(resized_image)
        temp = ' Differ             ' + str(int(self.parameters["DiffAmount"]*10)) + '%'
        self.differ_button = tk.Button(self.label_frame1, self.inactive_button_style, compound='left', image=self.diff_icon, text=temp, anchor='w', command=lambda: self.toggle_differ())
        self.differ_button.place(x=column1, y=37, width = 125, height = 26) 
        self.differ_button.bind("<MouseWheel>", self.change_differ_amount)        

        column2=column1+125+x_space
        # Mask top
        # Mask top-label
        img = Image.open('./rope/media/maskup.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.masktop_icon = ImageTk.PhotoImage(resized_image)        
        temp = ' Top Mask     ' + str(int(self.parameters["MaskTop"]*100.0/64.0)) + '%'
        
        self.top_blend_id = tk.Label(self.label_frame1, self.label_style, compound='left', image=self.masktop_icon, text=temp, anchor='w')
        self.top_blend_id.place(x=column2, y=8, width = 125, height = 26)
        self.top_blend_id.bind("<MouseWheel>", self.change_mask_top_amount) 

        # # Mask sides
        # # Mask sides-label
        # img = Image.open('./rope/media/maskside.png')
        # resized_image= img.resize((20,20), Image.ANTIALIAS)
        # self.maskside_icon = ImageTk.PhotoImage(resized_image)        
        # temp = ' Side Mask    ' + str(int(self.parameters["MaskSide"]*100.0/64.0)) + '%'
        # self.side_blend_id = tk.Label(self.label_frame1, self.label_style, compound='left', image=self.maskside_icon, text=temp, anchor='w')
        # self.side_blend_id.place(x=column2, y=37, width = 125, height = 26)
        # self.side_blend_id.bind("<MouseWheel>", self.change_mask_side_amount) 

        # # Mask blur
        # # Mask blur-label 
        img = Image.open('./rope/media/maskblur.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.maskblur_icon = ImageTk.PhotoImage(resized_image)  
        temp = ' Mask Blur    ' + str(int(self.parameters["MaskBlur"]*100.0/64.0)) + '%'
        self.mask_blur_id = tk.Label(self.label_frame1, self.label_style, compound='left', image=self.maskblur_icon, text=temp, anchor='w')
        self.mask_blur_id.place(x=column2, y=37, width = 125, height = 26)
        self.mask_blur_id.bind("<MouseWheel>", self.change_mask_blur_amount) 

        column3=column2+125+x_space
        # CLIP
        # CLIP-checkbox
        img = Image.open('./rope/media/CLIP.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.CLIP_icon = ImageTk.PhotoImage(resized_image)        
        temp = ' CLIP              ' + str(int(self.parameters["CLIPAmount"]*100)) + '%'
        self.CLIP_button = tk.Button(self.label_frame1, self.inactive_button_style, compound='left', image=self.CLIP_icon, text=temp, anchor='w', command=lambda: self.toggle_CLIP())
        self.CLIP_button.place(x=column3, y=8, width=125, height=26)
        self.CLIP_button.bind("<MouseWheel>", self.change_CLIP_amount)         

        # CLIP-entry
        self.CLIP_text = tk.Entry(self.label_frame1, relief='flat', bd=0, textvariable=self.parameters["CLIPText"])
        self.CLIP_text.place(x=column3, y=40, width = 125, height=20) 
        self.CLIP_text.bind("<Return>", lambda event: self.add_action_and_update_frame("parameters", self.parameters))

        column4=column3+125+x_space
        # # Occluder
        # # Occluder-checkbox
        img = Image.open('./rope/media/occluder.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.occluder_icon = ImageTk.PhotoImage(resized_image)        
        temp = ' Occluder'        
        self.occluder_button = tk.Button(self.label_frame1, self.inactive_button_style, compound='left', image=self.occluder_icon, text=temp, anchor='w', command=lambda: self.toggle_occluder())
        self.occluder_button.place(x=column4, y=8, width=125, height=26)

        # # Face Parser
        # # Face Parser-checkbox
        img = Image.open('./rope/media/parse.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.parser_icon = ImageTk.PhotoImage(resized_image)        
        temp = ' Mouth Parser'        
        self.parser_button = tk.Button(self.label_frame1, self.inactive_button_style, compound='left', image=self.parser_icon, text=temp, anchor='w', command=lambda: self.toggle_parser())
        self.parser_button.place(x=column4, y=37, width=125, height=26)        
        
        column5=column4+125+x_space
        # # Blur
        # # Blur-label
        img = Image.open('./rope/media/blur.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.blur_icon = ImageTk.PhotoImage(resized_image)        
        temp = ' Blur                   ' + str(int(self.parameters["BlurAmount"]*100.0/64.0)) + '%'
        self.blur_id = tk.Label(self.label_frame1, self.label_style, compound='left', image=self.blur_icon, text=temp, anchor='w')
        self.blur_id.place(x=column5, y=8, width = 125, height = 26)
        self.blur_id.bind("<MouseWheel>", self.change_blur_amount) 

        # # Face Threshhold
        # # Face Threshhold-label
        img = Image.open('./rope/media/thresh.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.threshhold_icon = ImageTk.PhotoImage(resized_image)        
        temp = ' Threshhold  ' + str(int(self.parameters["Threshhold"]*100)) + '%'
        self.threshhold_button = tk.Button(self.label_frame1, self.inactive_button_style, compound='left', image=self.threshhold_icon, text=temp, anchor='w', command=lambda: self.toggle_threshhold())
        self.threshhold_button.place(x=column5, y=37, width=125, height=26)
        self.threshhold_button.bind("<MouseWheel>", self.change_threshhold_amount)         

 ######## Target Faces           
        # Found Faces frame [1,0]
        self.found_faces_frame = tk.Frame( self, self.frame_style)
        self.found_faces_frame.grid( row = 2, column = 0, sticky='NEWS', pady = 2 )
        
        self.found_faces_frame.grid_columnconfigure( 0, minsize = 10 ) 
        self.found_faces_frame.grid_columnconfigure( 1, weight = 1 )         
        self.found_faces_frame.grid_rowconfigure( 0, weight = 0 )  
        
        # Button Canvas [0,0]
        self.found_faces_buttons_canvas = tk.Canvas( self.found_faces_frame, self.canvas_style1, height = 100, width = 112)
        self.found_faces_buttons_canvas.grid( row = 0, column = 0, )

        # Faces Load
        img = Image.open('./rope/media/tarface.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.target_faces_load_icon = ImageTk.PhotoImage(resized_image)
        self.found_faces_load_button = tk.Button(self.found_faces_buttons_canvas, self.inactive_button_style, image=self.target_faces_load_icon, compound='left', anchor='w', text=" Find", command=lambda: self.add_action_and_update_frame("find_faces", "current", False))
        self.found_faces_load_button.place(x=8, y=8, width = 96, height = 26)        
        
        # Faces Clear
        img = Image.open('./rope/media/tarfacedel.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.target_faces_del_icon = ImageTk.PhotoImage(resized_image)
        self.found_faces_clear_button = tk.Button(self.found_faces_buttons_canvas, self.inactive_button_style, image=self.target_faces_del_icon, compound='left', anchor='w', text=" Clear", command=lambda: self.add_action_and_update_frame("clear_faces", "current", False))
        self.found_faces_clear_button.place(x=8, y=37, width = 96, height = 26)    
        
        # Video Swap
        img = Image.open('./rope/media/swap.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.swap_icon = ImageTk.PhotoImage(resized_image)        
        self.video_swap = tk.Button( self.found_faces_buttons_canvas, self.inactive_button_style, image=self.swap_icon, compound='left', anchor='w', text=" Swap", command=lambda: self.toggle_swapper())
        self.video_swap.place(x=8, y=66, width = 96, height = 26)   
        
        # Faces Canvas [0,1]
        self.found_faces_canvas = tk.Canvas( self.found_faces_frame, self.canvas_style1, height = 100 )
        self.found_faces_canvas.grid( row = 0, column = 1, sticky='NEWS')
        self.found_faces_canvas.bind("<MouseWheel>", self.target_faces_mouse_wheel)
        self.found_faces_canvas.create_text(8, 45, anchor='w', fill='grey25', font=("Arial italic", 50), text=" Target Faces")
        
                # # Label
        # self.target_faces_id = tk.Canvas(self.found_faces_buttons_canvas, self.canvas_label_style)
        # self.target_faces_id.place(x=8, y=8, width = 20, height = 84) 
        # self.target_faces_id.create_text(8, 45, justify='center', fill='white', font=("Arial italic", 9), text="Target Faces", angle=90 )
    
                
 ######## Source Faces       
        # Source Faces frame [2,0]
        self.source_faces_frame = tk.Frame( self, self.frame_style)
        self.source_faces_frame.grid( row = 3, column = 0, sticky='NEWS', pady = 2 )

        self.source_faces_frame.grid_columnconfigure( 0, minsize = 10 ) 
        self.source_faces_frame.grid_columnconfigure( 1, weight = 1 )         
        self.source_faces_frame.grid_rowconfigure( 0, weight = 0 )  
        
        # Button Canvas [0,0]
        self.source_faces_buttons = []
        self.source_button_canvas = tk.Canvas( self.source_faces_frame, self.canvas_style1, height = 100, width = 112)
        self.source_button_canvas.grid( row = 0, column = 0, sticky='NEWS')
        
        # Load Source Faces
        img = Image.open('./rope/media/save.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.save_icon = ImageTk.PhotoImage(resized_image) 
        self.faces_filepath_button = tk.Button(self.source_button_canvas,  self.need_button_style, image=self.save_icon, compound='left', anchor='w', text="Source Faces", wraplength=120, command=lambda: self.select_faces_path())
        self.faces_filepath_button.place(x=8, y=8, width = 96, height = 26) 
        
        # Merged Embeddings Text
        self.merged_embedding_name = tk.StringVar()
        self.merged_embeddings_text = tk.Entry(self.source_button_canvas, relief='flat', bd=0, textvariable=self.merged_embedding_name)
        self.merged_embeddings_text.place(x=8, y=37, width = 96, height=20) 
        self.merged_embeddings_text.bind("<Return>", lambda event: self.save_selected_source_faces(self.merged_embedding_name)) 
        
        # Embedding remove
        img = Image.open('./rope/media/delemb.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.delemb_icon = ImageTk.PhotoImage(resized_image) 
        self.merged_embedding_remove_button = tk.Button(self.source_button_canvas, self.inactive_button_style, image=self.delemb_icon, compound='left', anchor='w', text=" Delete", command=lambda: self.delete_merged_embedding())
        self.merged_embedding_remove_button.place(x=8, y=66, width = 96, height = 26) 

        # Faces Canvas [0,1]
        self.source_faces_canvas = tk.Canvas( self.source_faces_frame, self.canvas_style1, height = 100)
        self.source_faces_canvas.grid( row = 0, column = 1, sticky='NEWS')
        self.source_faces_canvas.bind("<MouseWheel>", self.source_faces_mouse_wheel)
        self.source_faces_canvas.create_text(8, 45, anchor='w', fill='grey25', font=("Arial italic", 50), text=' Source Faces')


######### Target Videos
        # Target Video frame [3,0]
        self.target_videos_frame = tk.Frame( self, self.frame_style)
        self.target_videos_frame.grid( row = 4, column = 0, sticky='NEWS', pady = 2 )
        
        self.target_videos_frame.grid_columnconfigure( 0, minsize = 10 ) 
        self.target_videos_frame.grid_columnconfigure( 1, weight = 1 )         
        self.target_videos_frame.grid_rowconfigure( 0, weight = 0 )  
        
        # Button Canvas [0,0]
        self.target_videos_buttons = []
        self.target_button_canvas = tk.Canvas(self.target_videos_frame, self.canvas_style1, height = 100, width = 112)
        self.target_button_canvas.grid( row = 0, column = 0, sticky='NEWS')
        
        # # Videos Load
        # self.target_video_load_button = tk.Button(self.target_button_canvas, self.button_style1, text="Reload videos", command=lambda: self.populate_target_videos())
        # self.target_video_load_button.place(x=8, y=8, width = 84, height = 20)    

        # Target Videos Filepath
        self.video_filepath_button = tk.Button(self.target_button_canvas, self.need_button_style, image=self.save_icon, compound='left', anchor='w', text="Target Videos", wraplength=115, command=lambda: self.select_video_path())
        self.video_filepath_button.place(x=8, y=8, width = 96, height = 26) 
        
        # Video Canvas [0,1]
        self.target_video_canvas = tk.Canvas( self.target_videos_frame, self.canvas_style1, height = 100)
        self.target_video_canvas.grid( row = 0, column = 1, sticky='NEWS')
        self.target_video_canvas.bind("<MouseWheel>", self.target_videos_mouse_wheel)
        self.target_video_canvas.create_text(8, 45, anchor='w', fill='grey25', font=("Arial italic", 50), text=' Target Videos')

        column = 8
        
 ######### Options
        x_space = 40
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

        # Load Folders
        img = Image.open('./rope/media/save.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.load_folders_icon = ImageTk.PhotoImage(resized_image)   
        self.load_folders_button = tk.Button(self.program_options_label, self.need_button_style, compound='left', image=self.load_folders_icon, text=" Load Folders", anchor='w', command=lambda: self.load_all())
        self.load_folders_button.place(x=column, y=8, width = 125, height = 26) 
        
        column=column+125+x_space
        # Save Videos Filepath
        self.save_video_filepath_button = tk.Button(self.program_options_label, self.need_button_style, image=self.save_icon, compound='left', anchor='w', text="Saved Videos", wraplength=115, command=lambda: self.select_save_video_path())
        self.save_video_filepath_button.place(x=column, y=8, width = 96, height = 26)   
        
        column=column+125+x_space
        # Threads
        img = Image.open('./rope/media/threads.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.threads_icon = ImageTk.PhotoImage(resized_image)        
        temp = ' Threads           ' + str(self.num_threads)
        self.num_threads_id = tk.Label(self.program_options_label, self.label_style, compound='left', image=self.threads_icon, text=temp, anchor='w')
        self.num_threads_id.place(x=column, y=8, width = 125, height = 26)
        self.num_threads_id.bind("<MouseWheel>", self.change_threads_amount)    
        
        column=column+125+x_space
        # Video Quality
        img = Image.open('./rope/media/maskside.png')
        resized_image= img.resize((20,20), Image.ANTIALIAS)
        self.video_quality_icon = ImageTk.PhotoImage(resized_image)        
        temp = ' Video Quality     ' + str(self.video_quality)
        self.vid_qual_button = tk.Label(self.program_options_label, self.label_style, compound='left', image=self.video_quality_icon, text=temp, anchor='w')
        self.vid_qual_button.place(x=column, y=8, width = 125, height=26)
        self.vid_qual_button.bind("<MouseWheel>", self.change_video_quality)  


        
        # Status
        self.status_frame = tk.Frame( self, bg='grey20', height = 15)
        self.status_frame.grid( row = 6, column = 0, sticky='NEWS', pady = 2 )
        
        self.status_label = tk.Label(self.status_frame, fg="white", bg='grey20')
        self.status_label.pack()
        # self.status_label_text = tk.Label(self.status_frame, anchor="w", bg='grey75', text="Threads:")
        # self.status_label_text.place(x=100, y=8, width = 50, height=17)    
 
    def target_faces_mouse_wheel(self, event):
        self.found_faces_canvas.xview_scroll(1*int(event.delta/120.0), "units") 
   

    def source_faces_mouse_wheel(self, event):
        self.source_faces_canvas.xview_scroll(1*int(event.delta/120.0), "units")

   
    def target_videos_mouse_wheel(self, event):
        self.target_video_canvas.xview_scroll(1*int(event.delta/120.0), "units")


    def initialize_gui( self ):

        self.title("Rope - Crystal")
        # self.overrideredirect(True)
        self.configure(bg='grey10')
        self.resizable(width=True, height=True) 

        self.geometry('%dx%d+%d+%d' % (800, 1020, self.winfo_screenwidth()/2-400, self.winfo_screenheight()/2-510))

        self.grid_columnconfigure(0, weight = 1)  

        self.grid_rowconfigure(0, weight = 10)
        self.grid_rowconfigure(1, weight = 0)  
        self.grid_rowconfigure(2, weight = 0)  
        self.grid_rowconfigure(3, weight = 0)
        self.grid_rowconfigure(4, weight = 0)    
        self.grid_rowconfigure(5, weight = 0)  
        self.grid_rowconfigure(6, weight = 0) 


        self.add_action_and_update_frame("vid_qual",int(self.video_quality), False)
        self.add_action_and_update_frame("num_threads",int(self.num_threads), False)        
        self.add_action_and_update_frame("parameters", self.parameters, False)
        

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
                temp = self.json_dict["source videos"]
                temp_len = len(temp)
                temp = '...'+temp[temp_len-10:]
                 
                self.video_filepath_button.configure(self.inactive_button_style, text=temp) 
        
            self.json_dict["source faces"] = json_object["source faces"]
            if self.json_dict["source faces"]:
                temp = self.json_dict["source faces"]
                temp_len = len(temp)
                temp = '...'+temp[temp_len-10:]
                
                self.faces_filepath_button.configure(self.inactive_button_style, text=temp)
            
            self.json_dict["saved videos"] = json_object["saved videos"]
            if self.json_dict["saved videos"]:
                temp = self.json_dict["saved videos"]
                temp_len = len(temp)
                temp = '...'+temp[temp_len-10:]
                 
                self.save_video_filepath_button.configure(self.inactive_button_style, text=temp)
                self.add_action_and_update_frame("saved_video_path",self.json_dict["saved videos"], False)
            
            self.json_dict["threads"] = json_object["threads"]
            if self.json_dict["threads"]:
                temp = self.json_dict["threads"]
                self.num_threads = int(temp)
                    
                temp = ' Threads           ' + str(self.num_threads)
                self.num_threads_id.config(text=temp)        
         
                self.add_action_and_update_frame("num_threads",int(self.num_threads), False)
                
            
            
    def load_all(self):
        if not self.json_dict["source videos"] or not self.json_dict["source faces"]:
            print("Please set faces and videos folders first!")
            return
        
        self.add_action_and_update_frame("load_models", True, False)
        self.load_folders_button.configure(self.inactive_button_style, text=" Folders loaded!")
        
        
    def select_video_path(self):
         
        temp = self.json_dict["source videos"]
         
        self.json_dict["source videos"] = filedialog.askdirectory(title="Select Target Videos Folder", initialdir=temp)
        
        temp = self.json_dict["source videos"]
        temp_len = len(temp)
        temp = '...'+temp[temp_len-10:]
         
        self.video_filepath_button.configure(self.inactive_button_style, text=temp) 
        
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)
            
        self.populate_target_videos()
            
    def select_save_video_path(self):
        temp = self.json_dict["saved videos"]
        
        self.json_dict["saved videos"] = filedialog.askdirectory(title="Select Save Video Folder", initialdir=temp)
        
        temp = self.json_dict["saved videos"]
        temp_len = len(temp)
        temp = '...'+temp[temp_len-10:]
         
        self.save_video_filepath_button.configure(self.inactive_button_style, text=temp) 
        
        self.add_action_and_update_frame("saved_video_path",self.json_dict["saved videos"], False)
        
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)    

    def select_faces_path(self):
        temp = self.json_dict["source faces"]
        
        self.json_dict["source faces"] = filedialog.askdirectory(title="Select Source Faces Folder", initialdir=temp)
        
        temp = self.json_dict["source faces"]
        temp_len = len(temp)
        temp = '...'+temp[temp_len-10:]
        
        self.faces_filepath_button.configure(self.inactive_button_style, text=temp)
         
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)
        
        self.load_source_faces()
            
    def load_source_faces(self):
        if not self.faceapp_model:
            print("Load model first")
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
                self.add_action_and_update_frame("source_embeddings", self.source_faces, False)
        
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
                    crop = cv2.resize( crop, (82, 82))
                    

                    
                    found = False
                    # Test for existing simularities
                    for j in range(len(self.target_faces)):
                        sim = self.findCosineDistance(ret[i].embedding, self.target_faces[j]["Embedding"])
                        
                        if sim<self.parameters["Threshhold"]:
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

        # Set all other Source Face buttons to False 
        for i in range(len(self.source_faces)):      
            self.source_faces[i]["TKButton"].config(self.inactive_button_style)
            if i != button:
                self.source_faces[i]["ButtonState"] = False

        # Toggle the selected Source Face
        self.source_faces[button]["ButtonState"] = not self.source_faces[button]["ButtonState"]

        # Determine which target face is selected
        if self.target_faces:
            for i in range(len(self.target_faces)):
                if self.target_faces[i]["ButtonState"]:
                    
                    # Clear the assignments
                    self.target_faces[i]["SourceFaceAssignments"] = []
                    
                    # Append new assignment if new state is True
                    if self.source_faces[button]["ButtonState"]:
                        self.target_faces[i]["SourceFaceAssignments"].append(button)
                        self.source_faces[button]["TKButton"].config(self.button_highlight_style) 
                    
                    break        

            
        self.add_action_and_update_frame("target_faces", self.target_faces)
  
    def toggle_source_faces_buttons_state_shift(self, event, button):  
        
        # Toggle the selected Source Face
        self.source_faces[button]["ButtonState"] = not self.source_faces[button]["ButtonState"]
        
        if self.source_faces[button]["ButtonState"]:
            self.source_faces[button]["TKButton"].config(self.button_highlight_style)
        else:
            self.source_faces[button]["TKButton"].config(self.inactive_button_style)

        # If a target face is selected
        for i in range(len(self.target_faces)):
            if self.target_faces[i]["ButtonState"]:
            
                # Clear all of the assignments
                self.target_faces[i]["SourceFaceAssignments"] = []

                # Iterate through all Source faces
                for j in range(len(self.source_faces)):  
                    
                    # If the source face is active
                    if self.source_faces[j]["ButtonState"]:
                        self.target_faces[i]["SourceFaceAssignments"].append(j)

                break
   
        self.add_action_and_update_frame("target_faces", self.target_faces)
    
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
                crop = cv2.resize( crop, (82, 82))
                temp = [crop, video_file]
                videos.append(temp)

        for i in range(len(videos)):
            self.target_videos_buttons.append(tk.Button(self.target_video_canvas, self.inactive_button_style, height = 86, width = 86))
    
        for i in range(len(videos)):  
            rgb_video = Image.fromarray(videos[i][0])        
            self.target_videos.append(ImageTk.PhotoImage(image=rgb_video))            
            self.target_videos_buttons[i].config( image = self.target_videos[i],  command=lambda i=i: self.load_target_video(i, videos[i][1]))
            self.target_videos_buttons[i].bind("<MouseWheel>", self.target_videos_mouse_wheel)
            self.target_video_canvas.create_window(i*92, 8, window = self.target_videos_buttons[i], anchor='nw')
            
        self.target_video_canvas.configure(scrollregion = self.target_video_canvas.bbox("all"))

    def load_target_video(self, button, video_file):
        self.video_loaded = True
        self.add_action_and_update_frame("load_target_video", video_file, False)
        for i in range(len(self.target_videos_buttons)):
            self.target_videos_buttons[i].config(self.inactive_button_style)
        self.target_videos_buttons[button].config(self.button_highlight_style)
        
        if self.swap == True:
            self.toggle_swapper()
        if self.play_video == True:
            self.toggle_play_video()
        
        self.clear_faces()
            
    
    def set_image(self, image, requested):
        self.video_image = image[0]
        if not requested:
            self.set_slider_position(image[1])
    # @profile
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
                    self.add_action_and_update_frame("play_video", "stop", False)
                    self.video_play.config(self.inactive_button_style)
                else:
                    self.add_action_and_update_frame("play_video", "record", False)
                    self.video_play.config(self.active_button_style)
            else:
                self.add_action_and_update_frame("play_video", "play", False)
                self.video_play.config(self.active_button_style)
            
        else:
            self.add_action_and_update_frame("play_video", "stop", False)
            self.video_play.config(self.inactive_button_style)
            if self.rec_video:
                self.toggle_rec_video()
    
    
    
    def toggle_swapper(self):
        self.swap = not self.swap
        
        if not self.swap:
            self.video_swap.config(self.inactive_button_style)
        else:
            self.video_swap.config(self.active_button_style) 
        
        if not self.play_video:
            self.add_action_and_update_frame("swap", self.swap)
        else:
            self.add_action_and_update_frame("swap", self.swap, False)
            
    def toggle_rec_video(self):
        if not self.play_video:
            self.rec_video = not self.rec_video
                
            if self.rec_video == False:
                self.video_record.config(self.inactive_button_style)
            else:
                self.video_record.config(self.active_button_style, bg='red') 
                
        
            
            

        
    def set_faceapp_model(self, faceapp):
        self.faceapp_model = faceapp
        

        
    
    def add_action_and_update_frame(self, action, parameter, update_frame=True):
        
        # Get values for self.parameters
        if action == "parameters":
            parameter = {
                        "GFPGANState":              parameter["GFPGANState"],
                        "GFPGANAmount":             parameter["GFPGANAmount"],
                        "DiffState":                parameter["DiffState"],
                        "DiffAmount":               parameter["DiffAmount"],
                        "Threshhold":               parameter["Threshhold"],
                        "ThreshholdState":          parameter["ThreshholdState"],
                        "MaskTop":                  parameter["MaskTop"],
                        "MaskSide":                 parameter["MaskSide"],
                        "MaskBlur":                 parameter["MaskBlur"],
                        "OccluderState":            parameter["OccluderState"],
                        "CLIPState":                parameter["CLIPState"],
                        "CLIPText":                 parameter["CLIPText"].get(),
                        "CLIPAmount":               parameter["CLIPAmount"],
                        "FaceParserState":          parameter["FaceParserState"],
                        "BlurAmount":               parameter["BlurAmount"]
                        }
        
        # Send over action/parmeters tuple
        temp = [action, parameter]
        self.action_q.append(temp) 

        # If the video is not playing and update_frame is true
        if not self.play_video and update_frame:
            temp = ["set_video_position", self.video_slider.get()]
            self.action_q.append(temp)            


        
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
        
    def mouse_wheel(self, event):    
        if event.delta > 0: 
            self.video_slider.set(self.video_slider.get()+1)
            self.add_action_and_update_frame("set_video_position", self.video_slider.get(), False)
        else:
            self.video_slider.set(self.video_slider.get()-1)
            self.add_action_and_update_frame("set_video_position", self.video_slider.get(), False)
            
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

    def toggle_GFPGAN(self):
        self.parameters["GFPGANState"] = not self.parameters["GFPGANState"]
        
        if self.parameters["GFPGANState"]:
            self.GFPGAN_button.config(self.active_button_style)
        else:
            self.GFPGAN_button.config(self.inactive_button_style)
            
        self.add_action_and_update_frame("parameters", self.parameters)
        
    def change_GFPGAN_amount(self, event):        
        self.parameters["GFPGANAmount"] += (5*int(event.delta/120.0))
        if self.parameters["GFPGANAmount"] > 100:
            self.parameters["GFPGANAmount"] = 100
        if self.parameters["GFPGANAmount"] < 0 :
            self.parameters["GFPGANAmount"] = 0

        if self.parameters["GFPGANAmount"] >= 100:
            temp = '             ' + str(int(self.parameters["GFPGANAmount"])) + '%'
        else:
            temp = '               ' + str(int(self.parameters["GFPGANAmount"])) + '%' 
        

        self.GFPGAN_button.config(text=temp)
        
        self.add_action_and_update_frame("parameters", self.parameters)
        
    def toggle_differ(self):
        self.parameters["DiffState"] = not self.parameters["DiffState"]
        
        if self.parameters["DiffState"]:
            self.differ_button.config(self.active_button_style)
        else:
            self.differ_button.config(self.inactive_button_style)
            
        self.add_action_and_update_frame("parameters", self.parameters)
        
    def change_differ_amount(self, event):        
        self.parameters["DiffAmount"] += (0.5*int(event.delta/120.0))
        if self.parameters["DiffAmount"] > 10:
            self.parameters["DiffAmount"] = 10
        if self.parameters["DiffAmount"] < 0 :
            self.parameters["DiffAmount"] = 0
        
        if self.parameters["DiffAmount"] >= 10:
            temp = ' Differ           ' + str(int(self.parameters["DiffAmount"]*10)) + '%'
        else:
            temp = ' Differ             ' + str(int(self.parameters["DiffAmount"]*10)) + '%'       

        self.differ_button.config(text=temp)
        
        self.add_action_and_update_frame("parameters", self.parameters)

    def toggle_threshhold(self):
        self.parameters["ThreshholdState"] = not self.parameters["ThreshholdState"]
        
        if self.parameters["ThreshholdState"]:
            self.threshhold_button.config(self.active_button_style)            
        else:
            self.threshhold_button.config(self.inactive_button_style)
            
        self.add_action_and_update_frame("parameters", self.parameters)

    def change_threshhold_amount(self, event):        
        self.parameters["Threshhold"] += (0.01*int(event.delta/120.0))
        if self.parameters["Threshhold"] > 1:
            self.parameters["Threshhold"] = 1
        if self.parameters["Threshhold"] < 0 :
            self.parameters["Threshhold"] = 0
        
        if self.parameters["Threshhold"] >= 1:
            temp = ' Threshhold' + str(int(self.parameters["Threshhold"]*100)) + '%'
        else:
            temp = ' Threshhold  ' + str(int(self.parameters["Threshhold"]*100)) + '%'
            
        self.threshhold_button.config(text=temp)
        
        self.add_action_and_update_frame("parameters", self.parameters)  
        
    def change_mask_top_amount(self, event):     

        self.parameters["MaskTop"] += (1*int(event.delta/120.0))
        if self.parameters["MaskTop"] > 64:
            self.parameters["MaskTop"] = 64
        if self.parameters["MaskTop"] < 0 :
            self.parameters["MaskTop"] = 0
        
        if self.parameters["MaskTop"] >= 64:
            temp = ' Top Mask   ' + str(int(self.parameters["MaskTop"]*100.0/64.0)) + '%'
        else:
            temp = ' Top Mask     ' + str(int(self.parameters["MaskTop"]*100.0/64.0)) + '%'
            
        self.top_blend_id.config(text=temp)
        
        self.add_action_and_update_frame("parameters", self.parameters)         
        
    def change_mask_side_amount(self, event):     

        self.parameters["MaskSide"] += (1*int(event.delta/120.0))
        if self.parameters["MaskSide"] > 64:
            self.parameters["MaskSide"] = 64
        if self.parameters["MaskSide"] < 0 :
            self.parameters["MaskSide"] = 0
        
        if self.parameters["MaskSide"] >= 64:
            temp = ' Side Mask  ' + str(int(self.parameters["MaskSide"]*100.0/64.0)) + '%'
        else:
            temp = ' Side Mask    ' + str(int(self.parameters["MaskSide"]*100.0/64.0)) + '%'
            
        self.side_blend_id.config(text=temp)
        
        self.add_action_and_update_frame("parameters", self.parameters)      

    def change_mask_blur_amount(self, event):     

        self.parameters["MaskBlur"] += (1*int(event.delta/120.0))
        if self.parameters["MaskBlur"] > 30:
            self.parameters["MaskBlur"] = 30
        if self.parameters["MaskBlur"] < 0 :
            self.parameters["MaskBlur"] = 0
        
        temp_num = str(int(self.parameters["MaskBlur"]*100.0/30.0))
        temp_num_len = 4-len(temp_num)
        
        temp = ' Mask Blur  ' + ' '*temp_num_len + temp_num + '%'
            
        self.mask_blur_id.config(text=temp)
        
        self.add_action_and_update_frame("parameters", self.parameters)        
        
    def toggle_CLIP(self):
        self.parameters["CLIPState"] = not self.parameters["CLIPState"]
        
        if self.parameters["CLIPState"]:
            self.CLIP_button.config(self.active_button_style)            
        else:
            self.CLIP_button.config(self.inactive_button_style)
            
        self.add_action_and_update_frame("parameters", self.parameters)

    def change_CLIP_amount(self, event):        
        self.parameters["CLIPAmount"] += (0.01*int(event.delta/120.0))
        if self.parameters["CLIPAmount"] > 1:
            self.parameters["CLIPAmount"] = 1
        if self.parameters["CLIPAmount"] < 0 :
            self.parameters["CLIPAmount"] = 0
        
        if self.parameters["CLIPAmount"] >= 1:
            temp = ' CLIP            ' + str(int(self.parameters["CLIPAmount"]*100)) + '%'
        else:
            temp = ' CLIP              ' + str(int(self.parameters["CLIPAmount"]*100)) + '%'
            
        self.CLIP_button.config(text=temp)
        
        self.add_action_and_update_frame("parameters", self.parameters) 

    def toggle_occluder(self):
        self.parameters["OccluderState"] = not self.parameters["OccluderState"]
        
        if self.parameters["OccluderState"]:
            self.occluder_button.config(self.active_button_style)            
        else:
            self.occluder_button.config(self.inactive_button_style)
            
        self.add_action_and_update_frame("parameters", self.parameters)

    def toggle_parser(self):
        self.parameters["FaceParserState"] = not self.parameters["FaceParserState"]
        
        if self.parameters["FaceParserState"]:
            self.parser_button.config(self.active_button_style)            
        else:
            self.parser_button.config(self.inactive_button_style)
            
        self.add_action_and_update_frame("parameters", self.parameters)

    def change_blur_amount(self, event):     

        self.parameters["BlurAmount"] += (1*int(event.delta/120.0))
        if self.parameters["BlurAmount"] > 64:
            self.parameters["BlurAmount"] = 64
        if self.parameters["BlurAmount"] < 0 :
            self.parameters["BlurAmount"] = 0
        
        if self.parameters["BlurAmount"] >= 10:
            temp = ' Blur                 ' + str(int(self.parameters["BlurAmount"]*100.0/64.0)) + '%'
        else:
            temp = ' Blur                   ' + str(int(self.parameters["BlurAmount"]*100.0/64.0)) + '%'
            
        self.blur_id.config(text=temp)
        
        self.add_action_and_update_frame("parameters", self.parameters)   

    def change_video_quality(self, event): 
        self.video_quality += (1*int(event.delta/120.0))
        
        if self.video_quality > 50:
            self.video_quality = 50
        if self.video_quality < 0 :
            self.video_quality = 0
        
        temp = ' Video Quality     ' + str(self.video_quality)
            
        self.vid_qual_button.config(text=temp)        
 
        self.add_action_and_update_frame("vid_qual",int(self.video_quality), False)

    def change_threads_amount(self, event): 
        self.num_threads += (1*int(event.delta/120.0))
        
        if self.num_threads > 10:
            self.num_threads = 10
        if self.num_threads < 1:
            self.num_threads = 1
        
        temp = ' Threads           ' + str(self.num_threads)
            
        self.num_threads_id.config(text=temp)        
 
        self.add_action_and_update_frame("num_threads",int(self.num_threads), False)
        
        self.json_dict["threads"] = self.num_threads
        with open("data.json", "w") as outfile:
            json.dump(self.json_dict, outfile)

        
    # https://discord.gg/EcdVAFJzqp
