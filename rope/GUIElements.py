import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk

from  rope.Dicts import DEFAULT_DATA
import rope.Styles as style

#import inspect print(inspect.currentframe().f_back.f_code.co_name, 'resize_image')

class Separator_x():
    def __init__(self, parent, x, y): 
        self.parent = parent
        self.x = x
        self.y = y
        self.parent.update()
        self.blank = tk.PhotoImage()
        self.sep = tk.Label(self.parent, bg='#090909', image=self.blank, compound='c', border=0, width=self.parent.winfo_width(), height=1)
        self.sep.place(x=self.x, y=self.y)
        # self.parent.bind('<Configure>', self.update_sep_after_window_resize)
    
    # def update_sep_after_window_resize(self, event):
        # self.parent.update()
        # self.sep.configure(width=self.parent.winfo_width())
        
    def hide(self):
        self.sep.place_forget()

    def unhide(self):
        self.parent.update()
        self.sep.place(x=self.x, y=self.y)  
        self.sep.configure(width=self.parent.winfo_width())
   

class Separator_y():
    def __init__(self, parent, x, y): 
        self.parent = parent
        self.x = x
        self.y = y        
        self.parent.update()
        self.blank = tk.PhotoImage()
        self.sep = tk.Label(self.parent, bg='#090909', image=self.blank, compound='c', border=0, width=1, height=self.parent.winfo_height())
        self.sep.place(x=self.x, y=self.y) 
        # self.parent.bind('<Configure>', self.update_sep_after_window_resize)
    
    # def update_sep_after_window_resize(self, event):
        # self.parent.update()
        # self.sep.configure(height=self.parent.winfo_height())
        
    def hide(self):
        self.sep.place_forget()

    def unhide(self):
        self.parent.update()
        self.sep.place(x=self.x, y=self.y)  
        self.sep.configure(height=self.parent.winfo_height())          

class Text():
    def __init__(self, parent, text, style_level, x, y, width, height):
        self.blank = tk.PhotoImage()
        
        if style_level == 1:
            self.style = style.text_1       
        elif style_level == 2:
            self.style = style.text_2
        elif style_level == 3:
            self.style = style.text_3
            
        self.label = tk.Label(parent, self.style, image=self.blank, compound='c', text=text, anchor='w', width=width, height=height)
        self.label.place(x=x, y=y)
    
    def configure(self, text):
        self.label.configure(text=text)

class Scrollbar_y():
    def __init__(self, parent, child): 

        self.child = child

        self.trough_short_dim = 15
        self.trough_long_dim = []
        self.handle_short_dim = self.trough_short_dim*0.5
        
        self.top_of_handle = []
        self.middle_of_handle = []
        self.bottom_of_handle = []
        
        self.old_coord = 0
                
        # Child data
        self.child.bind('<Configure>', self.resize_scrollbar)

        # Set the canvas
        self.scrollbar_canvas = parent
        self.scrollbar_canvas.configure(width=self.trough_short_dim)
        self.scrollbar_canvas.bind("<MouseWheel>", self.scroll)
        self.scrollbar_canvas.bind("<ButtonPress-1>", self.scroll)
        self.scrollbar_canvas.bind("<B1-Motion>", self.scroll)     
        
        # Draw handle        
        self.resize_scrollbar(None)

    def resize_scrollbar(self, event): # on window updates
        self.child.update()
        self.child.configure(scrollregion=self.child.bbox("all"))
        
        # Reconfigure data
        self.trough_long_dim = self.child.winfo_height()
        self.scrollbar_canvas.delete('all')   
        self.scrollbar_canvas.configure(height=self.trough_long_dim)          
        
        # Redraw the scrollbar
        x1 = (self.trough_short_dim-self.handle_short_dim)/2
        x2 = self.trough_short_dim-x1
        y1 = self.child.yview()[0]*self.trough_long_dim
        y2 = self.child.yview()[1]*self.trough_long_dim
  
        self.middle_of_handle = self.scrollbar_canvas.create_rectangle(x1, y1, x2, y2, fill='grey25', outline='')

    def scroll(self, event):
        delta = 0
    
        # Get handle dimensions        
        handle_y1 = self.scrollbar_canvas.coords(self.middle_of_handle)[1]
        handle_y2 = self.scrollbar_canvas.coords(self.middle_of_handle)[3]
        handle_center = (handle_y2-handle_y1)/2 + handle_y1
        handle_length = handle_y2-handle_y1    

        if event.type == '38': # mousewheel
            delta = -int(event.delta/20.0)

        elif event.type == '4': # l-button press        
            # If the mouse coord is within the handle dont jump the handle
            if event.y > handle_y1 and event.y<handle_y2:
                self.old_coord = event.y
            else:
                self.old_coord = handle_center

            delta = event.y-self.old_coord
  
        elif event.type == '6': # l-button drag
            delta = event.y-self.old_coord
            
        # Do some bounding
        if handle_y1+delta<0:
            delta = -handle_y1
        elif handle_y2+delta>self.trough_long_dim:
            delta = self.trough_long_dim-handle_y2

        # update the scrollbar
        self.scrollbar_canvas.move(self.middle_of_handle, 0, delta)

        # Get the new handle postition to calculate the change for the child
        handle_y1 = self.scrollbar_canvas.coords(self.middle_of_handle)[1]

        # Move the child
        self.child.yview_moveto(handle_y1/self.trough_long_dim)
        
        self.old_coord = event.y
        
    def set(self, value):
        handle_y1 = self.scrollbar_canvas.coords(self.middle_of_handle)[1]
        handle_y2 = self.scrollbar_canvas.coords(self.middle_of_handle)[3]
        handle_center = (handle_y2-handle_y1)/2 + handle_y1
        
        coord_del = self.scrollbar_canvas.winfo_height()*value-handle_center
        self.old_coord = self.scrollbar_canvas.winfo_height()*value
        
        self.scrollbar_canvas.move(self.middle_of_handle, 0, coord_del)
  
    def hide(self):
        pass

    def unhide(self):
        pass   

class Timeline():
    def __init__(self, parent, widget, temp_toggle_swapper, add_action):  
        self.parent = parent
        self.add_action = add_action
        self.temp_toggle_swapper = temp_toggle_swapper

        self.frame_length = 0 
        self.height = 20
        self.counter_width = 40        

        self.entry_string = tk.StringVar()
        self.entry_string.set(0)

        self.last_position = 0
        
        # Widget variables
        self.max_ = 100#video_length 

        self.handle = []
        self.slider_left = []
        self.slider_right = []
           
        # Event trigget for window resize
        self.parent.bind('<Configure>', self.window_resize)

        # Add the Slider Canvas to the frame
        self.slider = tk.Canvas(self.parent, style.timeline_canvas, height=self.height)
        self.slider.place(x=0, y=0)
        self.slider.bind('<B1-Motion>', lambda e: self.update_timeline_handle(e, True))
        self.slider.bind('<ButtonPress-1>', lambda e: self.update_timeline_handle(e, True))
        self.slider.bind('<ButtonRelease-1>', lambda e: self.update_timeline_handle(e, True))
        self.slider.bind('<MouseWheel>', lambda e: self.update_timeline_handle(e, True))

        # Add the Entry to the frame
        self.entry_width = 40
        self.entry = tk.Entry(self.parent, style.entry_3, textvariable=self.entry_string)
        self.entry.bind('<Return>', lambda event: self.entry_input(event))      

    def draw_timeline(self):
        self.slider.delete('all')
        
        # Configure widths and placements
        self.slider.configure(width=self.frame_length)
        self.entry.place(x=self.parent.winfo_width()-self.counter_width, y=0)  

        # Draw the slider
        slider_pad = 20
        entry_pad = 20
        self.slider_left = slider_pad
        self.slider_right = self.frame_length-entry_pad-self.entry_width 
        slider_center = (self.height)/2

        line_loc = self.pos2coord(self.last_position)

        line_height = 8
        line_width = 1.5
        line_x1 = line_loc-line_width
        line_y1 = slider_center -line_height
        line_x2 = line_loc+line_width
        line_y2 = slider_center +line_height        

        
        trough_x1 = self.slider_left
        trough_y1 = slider_center-1
        trough_x2 = self.slider_right
        trough_y2 = slider_center+1  

        self.slider.create_rectangle(trough_x1, trough_y1, trough_x2, trough_y2, fill='#43474D', outline='')        
        self.handle = self.slider.create_rectangle(line_x1, line_y1, line_x2, line_y2, fill='#FFFFFF', outline='')    

    def coord2pos(self, coord):
        return float(coord-self.slider_left)*self.max_/(self.slider_right-self.slider_left)
        
    def pos2coord(self, pos):
        return float(float(pos)*(self.slider_right-self.slider_left)/self.max_ + self.slider_left)
  

    def update_timeline_handle(self, event, also_update_entry=False):
        requested = True

        if isinstance(event, float):
            position = event
            requested = False
        else:
            if event.type == '38': # mousewheel
                position = self.last_position+int(event.delta/120.0)
                
            elif event.type == '4': # l-button press
                x_coord = float(event.x)
                position = self.coord2pos(x_coord)
                
                # Turn off swapping
                self.temp_toggle_swapper('off')
                self.add_action("play_video", "stop")
                
            elif event.type == '5': # l-button release 
                x_coord = float(event.x)
                position = self.coord2pos(x_coord)       
                
                # Turn on swapping, if it was already on and request new frame
                self.temp_toggle_swapper('on')

            elif event.type == '6': # l-button drag
                x_coord = float(event.x)
                position = self.coord2pos(x_coord)

        # constrain mousewheel movement
        if position < 0: position = 0
        elif position > self.max_: position = self.max_

        # Find closest position increment
        position = round(position)    

        # moving sends many events, so only update when the next frame is reached
        if position != self.last_position:
            # Move handle to coordinate based on position
            self.slider.move(self.handle, self.pos2coord(position) - self.pos2coord(self.last_position), 0)
            
            if requested:
                self.add_action("get_requested_video_frame", position)          

            # Save for next time           
            self.last_position = position
            
            if also_update_entry:
                self.entry_string.set(str(position))

    def entry_input(self, event): 
    # event.char
        self.entry.update()
        try:
            input_num = float(self.entry_string.get())
            self.update_timeline_handle(input_num, False)
        except:
            return
    
    def set(self, value):
        self.update_timeline_handle(float(value), also_update_entry=True)

    def get(self):
        return int(self.last_position)
        
        
    def set_length(self, value):
        self.max_ = value
        self.update_timeline_handle(float(self.last_position), also_update_entry=True)

    def get_length(self):
        return int(self.max_)
        
    # Event when the window is resized
    def window_resize(self, event):
        self.parent.update()
        self.frame_length = self.parent.winfo_width()
        self.draw_timeline()
          


        
class Button():
    def __init__(self, parent, name, style_level, function, args, data_type, x, y, width=125, height=20):          
        self.default_data = DEFAULT_DATA
        self.name = name
        self.function = function
        self.args = args
        self.info = []
        self.state = []
        self.hold_state = []
        self.error = []
        self.data_type = data_type
        
        if style_level == 1:
            self.button_style = style.button_1        
        elif style_level == 2:
            self.button_style = style.button_2 
        elif style_level == 3:
            self.button_style = style.button_3


        # Add Icon
        if self.default_data[self.name+'Display'] == 'both':   
            img = Image.open(self.default_data[self.name+'IconOn'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_on = ImageTk.PhotoImage(resized_image)
            img = Image.open(self.default_data[self.name+'IconOff'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_off = ImageTk.PhotoImage(resized_image)
            img = Image.open(self.default_data[self.name+'IconHover'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_hover = ImageTk.PhotoImage(resized_image)
            
            text = ' '+self.default_data[self.name+'Text']
            
        elif self.default_data[self.name+'Display'] == 'icon':   
            img = Image.open(self.default_data[self.name+'IconOn'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_on = ImageTk.PhotoImage(resized_image)
            img = Image.open(self.default_data[self.name+'IconOff'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_off = ImageTk.PhotoImage(resized_image)
            img = Image.open(self.default_data[self.name+'IconHover'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_hover = ImageTk.PhotoImage(resized_image)            
            
            text = ''
            
        elif self.default_data[self.name+'Display'] == 'text':
            self.icon_on = tk.PhotoImage()
            self.icon_off = tk.PhotoImage()
            self.icon_hover = tk.PhotoImage()
            
            text = ' '+self.default_data[self.name+'Text']

        # Create Button and place
        self.button = tk.Button(parent, self.button_style, compound='left', text=text, anchor='w')
        self.button.configure(width=width, height=height)
        self.button.place(x=x, y=y)   

        self.button.bind("<Enter>", lambda event: self.on_enter())
        self.button.bind("<Leave>", lambda event: self.on_leave())     
        
        if self.function != None:
            if self.args != None:
                self.button.configure(command=lambda: self.function(self.args))
            else:
                self.button.configure(command=lambda: self.function())
                
        # Set inital state
        self.button.configure(image=self.icon_on) 
        
        if self.default_data[self.name+'State']:
            self.enable_button()

        else:
            self.disable_button()
    
    def add_info_frame(self, info):
        self.info = info
        
    
    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name+'InfoText'])
    
        if not self.state and not self.error:
            self.button.configure(image=self.icon_hover)    
            self.button.configure(fg='#B1B1B2') 

    def on_leave(self):
        if not self.state and not self.error:

            self.button.configure(image=self.icon_off)  
            self.button.configure(fg='#828282') 

    def enable_button(self):

        self.button.configure(image=self.icon_on)
        self.button.configure(fg='#FFFFFF') 
        self.state = True
        self.error = False
        
    def disable_button(self):

        self.button.configure(image=self.icon_off)
        self.button.configure(fg='#828282') 
        self.state = False
        self.error = False

    def toggle_button(self):
        self.state = not self.state
        
        if self.state:
            self.button.configure(image=self.icon_on)
            self.button.configure(fg='#FFFFFF') 
        else:
            self.button.configure(image=self.icon_off)
            self.button.configure(fg='#828282') 
       
    def temp_disable_button(self):
        self.hold_state = self.state
        self.state = False
        
    def temp_enable_button(self):
        self.state = self.hold_state

    def error_button(self):

        self.button.configure(image=self.icon_off)
        self.button.configure(fg='light goldenrod') 
        self.state = False 
        self.error = True
        
    def get(self):
        return self.state
        
    def set(self, value, request_frame=True):
        if value:
            self.enable_button()

        elif not value:
            self.disable_button()
        if request_frame:
            if self.function != None:
                if self.args != None:
                    self.function(self.args)
                else:
                    self.function()            
       
    def hide(self):
        pass
        
    def unhide(self):
        pass

    def get_data_type(self):
        return self.data_type
        
    def load_default(self):
        self.set(self.default_data[self.name+'State'])  

class TextSelection():
    def __init__(self, parent, name, display_text, style_level, function, argument, data_type, width, height, x, y, text_percent):
        self.blank = tk.PhotoImage()

        self.default_data = DEFAULT_DATA
        # Capture inputs as instance variables
        self.parent = parent
        self.name = name
        self.function = function
        self.argument = argument
        self.data_type = data_type
        self.width = width
        self.height = height
        self.style = []
        self.info = []

        if style_level == 3:
            self.frame_style = style.canvas_frame_label_3 
            self.text_style = style.text_3
            self.sel_off_style = style.text_selection_off_3
            self.sel_on_style = style.text_selection_on_3
        
        if style_level == 2:
            self.frame_style = style.canvas_frame_label_2
            self.text_style = style.text_2
            self.sel_off_style = style.text_selection_off_2
            self.sel_on_style = style.text_selection_on_2
        
        self.display_text = display_text+' '
        
        self.textselect_label = {}
        
        # Initial data
        self.selection = self.default_data[self.name+'Mode']
        
        # Frame to hold everything
        self.ts_frame = tk.Frame(self.parent, self.frame_style, width=self.width, height=self.height)
        self.ts_frame.place(x=x, y=y)
        self.ts_frame.bind("<Enter>", lambda event: self.on_enter())
        
        self.text_width = int(width*(1.0-text_percent)) 
        
        # Create the text on the left
        self.text_label = tk.Label(self.ts_frame, self.text_style, image=self.blank, compound='c', text=self.display_text, anchor='e', width=self.text_width, height=height)
        self.text_label.place(x=0, y=0)
        
        # Loop through the parameter modes, create a label 
        # Gotta find the size of the buttons according to the font
        self.font = tk.font.Font(family="Segoe UI", size=10, weight="normal")
        x_spacing = self.text_width + 10

        
        for mode in self.default_data[self.name+'Modes']:
            # Get size of text in pixels
            m_len = self.font.measure(mode)
            
            # Create a label with the text
            self.textselect_label[mode] = tk.Label(self.ts_frame, self.sel_off_style, text=mode, image=self.blank, compound='c', anchor='c', width=m_len, height=height)
            self.textselect_label[mode].place(x=x_spacing, y=0)
            self.textselect_label[mode].bind("<ButtonRelease-1>", lambda event, mode=mode: self.select_ui_text_selection(mode))
            
            # Initial value
            if mode==self.selection:
                self.textselect_label[mode].configure(self.sel_on_style)
            
            x_spacing = x_spacing + m_len+10

    def select_ui_text_selection(self, selection, request_frame=True):
        # Loop over all of the Modes
        for mode in self.default_data[self.name+'Modes']:

            # If the Mode has been selected
            if mode==selection:      
                # Set state to true
                self.textselect_label[mode].configure(self.sel_on_style)
                self.selection = mode
                if request_frame:
                    self.function(self.argument, self.name)

            else:
                self.textselect_label[mode].configure(self.sel_off_style)  

    def add_info_frame(self, info):
        self.info = info 
        
    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name+'InfoText'])
 
    def get(self):
        return self.selection
    
    def set(self, value, request_frame=True):
        self.select_ui_text_selection(value, request_frame)
        
    def hide(self):
        pass
        
    def unhide(self):
        pass 
        
    def get_data_type(self):
        return self.data_type  

    def load_default(self):
        self.set(self.default_data[self.name+'Mode'])           
 

class Switch2():
    def __init__(self, parent, name, display_text, style_level, function, argument, width, height, x, y):
        self.blank = tk.PhotoImage()
        self.default_data = DEFAULT_DATA
        # Capture inputs as instance variables
        self.parent = parent
        self.name = name
        self.function = function
        self.argument = argument
        self.data_type = argument
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.style = []
        self.info = []
        
        # Initial Value
        self.state = self.default_data[name+'State']
        
        if style_level == 3:
            self.frame_style = style.canvas_frame_label_3 
            self.text_style = style.text_3
            self.entry_style = style.entry_3 
      
        self.display_text = display_text
        # Load Icons
        self.img = Image.open(style.icon['IconOff'])
        self.img = self.img.resize((40,40), Image.ANTIALIAS)
        self.icon_off = ImageTk.PhotoImage(self.img)    
        
        self.img = Image.open(style.icon['IconOn'])
        self.img = self.img.resize((40,40), Image.ANTIALIAS)
        self.icon_on = ImageTk.PhotoImage(self.img) 

        # Frame to hold everything
        self.switch_frame = tk.Frame(self.parent, self.frame_style, width=self.width, height=self.height)
        self.switch_frame.place(x=self.x, y=self.y)
        self.switch_frame.bind("<Enter>", lambda event: self.on_enter())
 

        toggle_width = 40
        text_width = self.width-toggle_width
        
        # Toggle Switch
        self.switch = tk.Label(self.switch_frame, style.parameter_switch_3, image=self.icon_off, width=toggle_width, height=self.height)
        if self.state:
            self.switch.configure(image=self.icon_on)
        self.switch.place(x=0, y=2)
        self.switch.bind("<ButtonRelease-1>", lambda event: self.toggle_switch(event))
        
        # Text
        self.switch_text = tk.Label(self.switch_frame, style.parameter_switch_3, image=self.blank, compound='right', text=self.display_text, anchor='w', width=text_width, height=height-2)
        self.switch_text.place(x=50, y=0)

    def toggle_switch(self, event, set_value=None, request_frame=True):
        # flip state
        if set_value==None:
            self.state = not self.state
        else:
            self.state = set_value
            
        if self.state:
            self.switch.configure(image=self.icon_on)
            
        else:
            self.switch.configure(image=self.icon_off)

        if request_frame:
            self.function(self.argument, self.name, use_markers=False)
    
    def add_info_frame(self, info):
        self.info = info
        
    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name+'InfoText'])
     
    
    def hide(self):
        self.switch_frame.place_forget()
        self.switch.place_forget()
        self.switch_text.place_forget()
        
    def unhide(self):
        self.switch_frame.place(x=self.x, y=self.y)    
        self.switch.place(x=0, y=2)
        self.switch_text.place(x=50, y=0)
        
    def set(self, value, request_frame=True):
        self.toggle_switch(None, value, request_frame)

    def get(self):
        return self.state   

    def get_data_type(self):
        return self.data_type 

    def load_default(self):
        self.set(self.default_data[self.name+'State'])        
                
class Slider2():
    def __init__(self, parent, name, display_text, style_level, function, argument, width, height, x, y, slider_percent):

        # self.constants = CONSTANTS
        self.default_data = DEFAULT_DATA
        self.blank = tk.PhotoImage() 
        
        # Capture inputs as instance variables
        self.parent = parent
        self.name = name
        self.function = function
        self.data_type = argument
        self.x = x
        self.y = y
        self.slider_percent = slider_percent
        self.width = width
        self.height = height
        self.info = []
        
        # Initial Value
        self.amount = self.default_data[name+'Amount']
        
        if style_level == 1:
            self.frame_style = style.canvas_frame_label_1 
            self.text_style = style.text_1
            self.entry_style = style.entry_3          

        elif style_level == 3:
            self.frame_style = style.canvas_frame_label_3 
            self.text_style = style.text_3
            self.entry_style = style.entry_3    
        
        # UI-controlled variables        
        self.entry_string = tk.StringVar()
        self.entry_string.set(self.amount)        
        
        # Widget variables
        self.min_ = self.default_data[name+'Min']
        self.max_ = self.default_data[name+'Max'] 
        self.inc_ = self.default_data[name+'Inc']   
        self.display_text = display_text+' '         

        # Set up spacing
        # |----------------------|slider_pad|-slider-|entry_pad|-|
        # |---1-slider_percent---|---slider_percent---|
        # |--------------------width------------------|        
        
        # Create a frame to hold it all
        self.frame_x = x
        self.frame_y = y
        self.frame_width = width
        self.frame_height = height
    
        self.frame = tk.Frame(self.parent, self.frame_style, width=self.frame_width, height=self.frame_height)
        self.frame.place(x=self.frame_x, y=self.frame_y)
        self.frame.bind("<Enter>", lambda event: self.on_enter())


        # Add the slider Label text to the frame
        self.txt_label_x = 0
        self.txt_label_y = 0
        self.txt_label_width = int(width*(1.0-slider_percent))        

        self.label = tk.Label(self.frame, self.text_style, image=self.blank, compound='c', text=self.display_text, anchor='e', width=self.txt_label_width, height=self.height)
        self.label.place(x=self.txt_label_x, y=self.txt_label_y)
        
        # Add the Slider Canvas to the frame
        self.slider_canvas_x = self.txt_label_width
        self.slider_canvas_y = 0
        self.slider_canvas_width = width-self.txt_label_width
        
        self.slider = tk.Canvas(self.frame, self.frame_style, width=self.slider_canvas_width, height=self.height)
        self.slider.place(x=self.slider_canvas_x, y=self.slider_canvas_y)
        self.slider.bind('<B1-Motion>', lambda e: self.update_handle(e, True))  
        self.slider.bind('<MouseWheel>', lambda e: self.update_handle(e, True))         

        # Add the Entry to the frame
        self.entry_width = 60
        self.entry_x = self.frame_width-self.entry_width 
        self.entry_y = 0

        self.entry = tk.Entry(self.frame, self.entry_style, textvariable=self.entry_string)
        self.entry.place(x=self.entry_x, y=self.entry_y)
        self.entry.bind('<Return>', lambda event: self.entry_input(event))      

        # Draw the slider
        self.slider_pad = 20
        self.entry_pad = 20
        self.slider_left = self.slider_pad
        self.slider_right = self.slider_canvas_width-self.entry_pad-self.entry_width 
        self.slider_center = (self.height+1)/2

        self.oval_loc = self.pos2coord(self.amount)
        self.oval_radius = 5
        self.oval_x1 = self.oval_loc-self.oval_radius
        self.oval_y1 = self.slider_center-self.oval_radius
        self.oval_x2 = self.oval_loc+self.oval_radius
        self.oval_y2 = self.slider_center+self.oval_radius
        
        self.trough_x1 = self.slider_left
        self.trough_y1 = self.slider_center-2
        self.trough_x2 = self.slider_right
        self.trough_y2 = self.slider_center+2  

        self.slider.create_rectangle(self.trough_x1, self.trough_y1, self.trough_x2, self.trough_y2, fill='#1F1F1F', outline='')        
        self.handle = self.slider.create_oval(self.oval_x1, self.oval_y1, self.oval_x2, self.oval_y2, fill='#919191', outline='')
        
    def coord2pos(self, coord):
        return float((coord-self.slider_left)*(self.max_-self.min_)/(self.slider_right-self.slider_left) + self.min_)
        
    def pos2coord(self, pos):
        return float((float(pos)-self.min_)*(self.slider_right-self.slider_left)/(self.max_-self.min_) + self.slider_left)

    def update_handle(self, event, also_update_entry=False, request_frame=True):
        if isinstance(event, float):
            position = event

        elif event.type == '38':
            position = self.amount+self.inc_*int(event.delta/120.0)

        elif event.type == '6':
            x_coord = float(event.x)
            position = self.coord2pos(x_coord)
            
        # constrain mousewheel movement
        if position < self.min_: position = self.min_
        elif position > self.max_: position = self.max_
        
        # Find closest position increment
        position_inc = round((position-self.min_) / self.inc_)      
        position = (position_inc * self.inc_)+self.min_

        # moving sends many events, so only update when the next frame is reached
        if position != self.amount:
            # Move handle to coordinate based on position
            self.slider.move(self.handle, self.pos2coord(position) - self.pos2coord(self.amount), 0)

            # Save for next time           
            self.amount = position
            
            if also_update_entry:
                self.entry_string.set(str(position))
            
            if request_frame:
                self.function(self.data_type, self.name, use_markers=False)
                
            # return True
        # return False
    
    def add_info_frame(self, info):
        self.info = info
    
    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name+'InfoText'])
        
    def entry_input(self, event): 
    # event.char
        self.entry.update()
        try:
            input_num = float(self.entry_string.get())
            self.update_handle(input_num, False)
        except:
            return
    
    def set(self, value, request_frame=True):
        self.update_handle(float(value), True)

    def get(self):
        return self.amount
        
    def hide(self):
        self.frame.place_forget()
        self.label.place_forget()
        self.slider.place_forget()
        self.entry.place_forget()
        
    def unhide(self):
        self.frame.place(x=self.frame_x, y=self.frame_y)    
        self.label.place(x=self.txt_label_x, y=self.txt_label_y)
        self.slider.place(x=self.slider_canvas_x, y=self.slider_canvas_y)    
        self.entry.place(x=self.entry_x, y=self.entry_y)  
            
    # def save_to_file(self, filename, data):
        # with open(filename, 'w') as outfile:
            # json.dump(data, outfile)
            
    def get_data_type(self):
        return self.data_type 

    def load_default(self):
        self.set(self.default_data[self.name+'Amount'])
            
class Text_Entry():
    def __init__(self, parent, name, display_text, style_level, function, data_type, width, height, x, y, text_percent):
        self.blank = tk.PhotoImage()

        self.default_data = DEFAULT_DATA
        # Capture inputs as instance variables
        self.parent = parent
        self.name = name
        self.function = function
        self.data_type = data_type
        self.width = width
        self.height = height
        self.style = []
        self.info = []

        if style_level == 3:
            self.frame_style = style.canvas_frame_label_3 
            self.text_style = style.text_3
            self.sel_off_style = style.text_selection_off_3
            self.sel_on_style = style.text_selection_on_3
        
        if style_level == 2:
            self.frame_style = style.canvas_frame_label_2
            self.text_style = style.text_2
            self.sel_off_style = style.text_selection_off_2
            self.sel_on_style = style.text_selection_on_2
        
        self.display_text = display_text+' '
        
     
        # Initial data
        self.entry_text = tk.StringVar()
        self.entry_text.set(self.default_data[self.name])
        
        # Frame to hold everything
        self.ts_frame = tk.Frame(self.parent, self.frame_style, width=self.width, height=self.height)
        self.ts_frame.place(x=x, y=y)
        self.ts_frame.bind("<Enter>", lambda event: self.on_enter())
        
        self.text_width = int(width*(1.0-text_percent))
        
        # Create the text on the left
        self.text_label = tk.Label(self.ts_frame, self.text_style, image=self.blank, compound='c', text=self.display_text, anchor='e', width=self.text_width, height=height)
        self.text_label.place(x=0, y=0)
        
        
        
        self.entry = tk.Entry(self.ts_frame, style.entry_2, textvariable=self.entry_text)
        self.entry.place(x=self.text_width+20, y=0, width = self.width-self.text_width-50, height=15) 
        self.entry.bind("<Return>", lambda event: self.send_text(self.entry_text.get())) 
        
    def send_text(self, text):
        self.function(self.data_type, self.name, use_markers=False)

    def add_info_frame(self, info):
        self.info = info 
        
    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name+'InfoText'])
 
    def get(self):
        return self.entry_text.get()
    
    def set(self, value, request_frame=True):
        pass
        # self.select_ui_text_selection(value, request_frame)
        
    def hide(self):
        pass
        
    def unhide(self):
        pass 
        
    def get_data_type(self):
        return self.data_type  

    def load_default(self):
        pass
        # self.set(self.default_data[self.name+'Mode']) 
        
class VRAM_Indicator():
    def __init__(self, parent, style_level, width, height, x, y):
        self.parent = parent
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.blank = tk.PhotoImage()
        
        self.used = 0
        self.total = 1
        
        if style_level == 3:
            self.frame_style = style.canvas_frame_label_3 
            self.text_style = style.text_3
            self.sel_off_style = style.text_selection_off_3
            self.sel_on_style = style.text_selection_on_3
        
        if style_level == 2:
            self.frame_style = style.canvas_frame_label_2
            self.text_style = style.text_2
            self.sel_off_style = style.text_selection_off_2
            self.sel_on_style = style.text_selection_on_2
        
        if style_level == 1:
            self.frame_style = style.canvas_frame_label_1

        self.frame = tk.Frame(self.parent, self.frame_style, width=self.width, height=self.height)
        self.frame.place(x=self.x, y=self.y)
        
        self.label_name = tk.Label(self.frame, self.frame_style, image=self.blank, compound='c', fg='#b1b1b2', font=("Segoe UI", 9), width=50, text='VRAM', height=self.height)
        self.label_name.place(x=0, y=0)

        
        # self.label_value = tk.Label(self.frame, self.frame_style, bg='yellow', image=self.blank, compound='c', fg='#D0D0D0', font=("Segoe UI", 9), justify='right', width=100, text='VRAM', height=self.height)
        # self.label_value.place(x=200, y=0)
        
        
        self.canvas = tk.Canvas(self.frame, self.frame_style, highlightthickness =2, highlightbackground='#b1b1b2', width=self.width-60, height=self.height-4)
        self.canvas.place(x=50, y=0)
        
    def update_display(self):
        self.canvas.delete('all')
        width = self.canvas.winfo_width()

        try:
            ratio = self.used/self.total
        except ZeroDivisionError:
            ratio = 1
        
        if ratio>0.9:
            color = '#d10303'
        else:
            color = '#b1b1b2'
        width = ratio*width
        
        self.canvas.create_rectangle(0, 0, width, self.height, fill=color)
        
        # text = str(self.used)+' / '+str(self.total)+' MB'        
        # self.label_value.configure(text=text)
    
    def set(self, used, total):
        self.used = used
        self.total = total
        
        self.update_display()
        
    def hide(self):
        pass
 
    def unhide(self):
        pass  
        