
import cv2
import numpy as np
from skimage import transform as trans
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from numpy.linalg import norm as l2norm
import onnxruntime
import onnx
from itertools import product as product
import subprocess as sp
onnxruntime.set_default_logger_severity(4)

class Models():
    def __init__(self): 
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32) 
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.retinaface_model = []
        self.yoloface_model = []
        self.scrdf_model = []
        self.resnet50_model, self.anchors  = [], []
        
        self.insight106_model = []

        self.recognition_model = []
        self.swapper_model = []

        self.emap = []
        self.GFPGAN_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.codeformer_model = []
        
        self.occluder_model = []
        self.faceparser_model = []
        
        self.syncvec = torch.empty((1,1), dtype=torch.float32, device='cuda:0')
        
    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_total = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
        
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        
        memory_used = memory_total[0] - memory_free[0]
        
        return memory_used, memory_total[0]     

    def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5):
        kpss = []
        
        if detect_mode=='Retinaface':
            if not self.retinaface_model:
                self.retinaface_model = onnxruntime.InferenceSession('.\models\det_10g.onnx', providers=self.providers)
                
            kpss = self.detect_retinaface(img, max_num=max_num, score=score)
            
        elif detect_mode=='SCRDF':
            if not self.scrdf_model:
                self.scrdf_model = onnxruntime.InferenceSession('.\models\scrfd_2.5g_bnkps.onnx', providers=self.providers)
                
            kpss = self.detect_scrdf(img, max_num=max_num, score=score)
            
        
        elif detect_mode=='Yolov8':
            if not self.yoloface_model:
                self.yoloface_model = onnxruntime.InferenceSession('.\models\yoloface_8n.onnx', providers=self.providers)
        
            kpss = self.detect_yoloface(img, max_num=max_num, score=score)
        
        return kpss
        
    def run_align(self, img):
        points = []

        if not self.insight106_model:
            self.insight106_model = onnxruntime.InferenceSession('.\models\2d106det.onnx', providers=self.providers)
            
        points = self.detect_insight106(img)   
    
    def delete_models(self):

        self.yoloface_model = []
        self.scrdf_model = []
        self.resnet50_model=[]
        self.insight106_model = []
        self.GFPGAN_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.codeformer_model = []
        self.occluder_model = []
        self.faceparser_model = []
            
    def run_recognize(self, img, kps):
        if not self.recognition_model:
            self.recognition_model = onnxruntime.InferenceSession('.\models\w600k_r50.onnx', providers=self.providers)
        
        embedding, cropped_image = self.recognize(img, kps)
        return embedding, cropped_image
    

    def calc_swapper_latent(self, source_embedding):
        if not self.swapper_model:
            graph = onnx.load("./models/inswapper_128.fp16.onnx").graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
        
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        return latent
        
    def run_swapper(self, image, embedding, output):
        if not self.swapper_model:
            # cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}            
            # sess_options = onnxruntime.SessionOptions()
            # sess_options.enable_cpu_mem_arena = False            
            
            # self.swapper_model = onnxruntime.InferenceSession( "./models/inswapper_128.fp16.onnx", sess_options, providers=[('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider'])
            
            self.swapper_model = onnxruntime.InferenceSession( "./models/inswapper_128.fp16.onnx", providers=self.providers)

        io_binding = self.swapper_model.io_binding()
        io_binding.bind_input(name='target', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.swapper_model.run_with_iobinding(io_binding)

    def run_GFPGAN(self, image, output):
        if not self.GFPGAN_model:
            # cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}
            # sess_options = onnxruntime.SessionOptions()
            # sess_options.enable_cpu_mem_arena = False

            # self.GFPGAN_model = onnxruntime.InferenceSession( "./models/GFPGANv1.4.onnx", sess_options, providers=[("CUDAExecutionProvider", cuda_options), 'CPUExecutionProvider'])
            
            self.GFPGAN_model = onnxruntime.InferenceSession( "./models/GFPGANv1.4.onnx", providers=self.providers)

        io_binding = self.GFPGAN_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())  
                
        self.syncvec.cpu()          
        self.GFPGAN_model.run_with_iobinding(io_binding)                

    
    def run_GPEN_512(self, image, output):
        if not self.GPEN_512_model:
            self.GPEN_512_model = onnxruntime.InferenceSession( "./models/GPEN-BFR-512.onnx", providers=self.providers)
 
        io_binding = self.GPEN_512_model.io_binding() 
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())            
        
        self.syncvec.cpu()          
        self.GPEN_512_model.run_with_iobinding(io_binding)   

    def run_GPEN_256(self, image, output):
        if not self.GPEN_256_model:
            self.GPEN_256_model = onnxruntime.InferenceSession( "./models/GPEN-BFR-256.onnx", providers=self.providers)

        io_binding = self.GPEN_256_model.io_binding()         
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())
        
        self.syncvec.cpu()          
        self.GPEN_256_model.run_with_iobinding(io_binding) 

    def run_codeformer(self, image, output):   
        if not self.codeformer_model:    
            self.codeformer_model = onnxruntime.InferenceSession( "./models/codeformer_fp16.onnx", providers=self.providers)

        io_binding = self.codeformer_model.io_binding() 
        io_binding.bind_input(name='x', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        w = np.array([0.9], dtype=np.double)
        io_binding.bind_cpu_input('w', w)
        io_binding.bind_output(name='y', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())
        
        self.syncvec.cpu()          
        self.codeformer_model.run_with_iobinding(io_binding)        

    def run_occluder(self, image, output):    
        if not self.occluder_model:
            self.occluder_model = onnxruntime.InferenceSession("./models/occluder.onnx", providers=self.providers)

        io_binding = self.occluder_model.io_binding()            
        io_binding.bind_input(name='img', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())   

        # torch.cuda.synchronize('cuda')  
        self.syncvec.cpu()         
        self.occluder_model.run_with_iobinding(io_binding)  


    def run_faceparser(self, image, output):    
        if not self.faceparser_model:
            self.faceparser_model = onnxruntime.InferenceSession("./models/faceparser_fp16.onnx", providers=self.providers)

        io_binding = self.faceparser_model.io_binding()            
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='out', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=output.data_ptr())   

        # torch.cuda.synchronize('cuda')  
        self.syncvec.cpu()         
        self.faceparser_model.run_with_iobinding(io_binding)       
    
        
    def detect_retinaface(self, img, max_num, score):
        # Resize image to fit within the input_size
        input_size = (640, 640)
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
        

        
        io_binding = self.retinaface_model.io_binding() 
        io_binding.bind_input(name='input.1', device_type='cuda', device_id=0, element_type=np.float32,  shape=det_img.size(), buffer_ptr=det_img.data_ptr())

        io_binding.bind_output('448', 'cuda') 
        io_binding.bind_output('471', 'cuda') 
        io_binding.bind_output('494', 'cuda') 
        io_binding.bind_output('451', 'cuda') 
        io_binding.bind_output('474', 'cuda') 
        io_binding.bind_output('497', 'cuda') 
        io_binding.bind_output('454', 'cuda') 
        io_binding.bind_output('477', 'cuda') 
        io_binding.bind_output('500', 'cuda') 
        
        
        # Sync and run model
        self.syncvec.cpu()        
        self.retinaface_model.run_with_iobinding(io_binding)
        
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
            
            pos_inds = np.where(scores>=score)[0]

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
        
    def detect_scrdf(self, img, max_num, score):
        # Resize image to fit within the input_size
        input_size = (640, 640)
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
        input_name = self.scrdf_model.get_inputs()[0].name
        
        outputs = self.scrdf_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        
        io_binding = self.scrdf_model.io_binding() 
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=det_img.size(), buffer_ptr=det_img.data_ptr())
        
        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda') 
        
        # Sync and run model
        syncvec = self.syncvec.cpu()        
        self.scrdf_model.run_with_iobinding(io_binding)
        
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
            
            pos_inds = np.where(scores>=score)[0]

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
        
    def detect_yoloface(self, img, max_num, score):

        height = img.size(dim=1)
        width = img.size(dim=2)
        length = max((height, width))

        image = torch.zeros((length, length, 3), dtype=torch.uint8, device='cuda')
        img = img.permute(1,2,0)
        
        image[0:height, 0:width] = img
        scale = length/640        
        image = torch.div(image, 255.0)
        
        t640 = v2.Resize((640, 640), antialias=False)
        image = image.permute(2, 0, 1)
        image = t640(image)

        image = torch.unsqueeze(image, 0).contiguous()

        io_binding = self.yoloface_model.io_binding() 
        io_binding.bind_input(name='images', device_type='cuda', device_id=0, element_type=np.float32,  shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output('output0', 'cuda') 
        
        # Sync and run model
        self.syncvec.cpu()     
        self.yoloface_model.run_with_iobinding(io_binding)
        
        net_outs = io_binding.copy_outputs_to_cpu()        

        outputs = np.squeeze(net_outs).T

        bbox_raw, score_raw, kps_raw = np.split(outputs, [4, 5], axis=1)

        bbox_list = []
        score_list = []
        kps_list = []
        keep_indices = np.where(score_raw > score)[0]

        if keep_indices.any():
            bbox_raw, kps_raw, score_raw = bbox_raw[keep_indices], kps_raw[keep_indices], score_raw[keep_indices]
            for bbox in bbox_raw:
                bbox_list.append(np.array([(bbox[0]-bbox[2]/2), (bbox[1]-bbox[3]/2), (bbox[0]+bbox[2]/2), (bbox[1]+bbox[3]/2)]))
            kps_raw = kps_raw * scale

            for kps in kps_raw:
                indexes = np.arange(0, len(kps), 3)
                temp_kps = []
                for index in indexes:
                    temp_kps.append([kps[index], kps[index + 1]])
                kps_list.append(np.array(temp_kps))
            score_list = score_raw.ravel().tolist()        

        result_boxes = cv2.dnn.NMSBoxes(bbox_list, score_list, 0.25, 0.45, 0.5)
        
        result = []
        for r in result_boxes:
            if r==max_num: 
                break
            result.append(kps_list[r])
            
        return np.array(result)
   
    def run_insight106(self, img):
        height = img.size(dim=1)
        width = img.size(dim=2)
        length = max((height, width))

        image = torch.zeros((length, length, 3), dtype=torch.uint8, device='cuda')
        img = img.permute(1,2,0)
        
        image[0:height, 0:width] = img
        scale = length/192       
        image = torch.div(image, 255.0)
        
        t192 = v2.Resize((192, 192), antialias=False)
        image = image.permute(2, 0, 1)
        image = t192(image)

        image = torch.unsqueeze(image, 0).contiguous()

        io_binding = self.insight106_model.io_binding() 
        io_binding.bind_input(name='data', device_type='cuda', device_id=0, element_type=np.float32,  shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output('fc1', 'cuda') 
        
        # Sync and run model
        self.syncvec.cpu()     
        self.insight106_model.run_with_iobinding(io_binding)
        
        net_outs = io_binding.copy_outputs_to_cpu() 
        
    def recognize(self, img, face_kps):
        # Find transform 
        tform = trans.SimilarityTransform()
        tform.estimate(face_kps, self.arcface_dst)

        # Transform
        img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) ) 
        img = v2.functional.crop(img, 0,0, 112, 112)

        # Switch to BGR and normalize
        img = img.permute(1,2,0) #112,112,3   
        cropped_image = img
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
        self.syncvec.cpu()
        self.recognition_model.run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image
        
    def resnet50(self, image, score=.5):   
        if not self.resnet50_model:
            self.resnet50_model = onnxruntime.InferenceSession("./models/res50.onnx", providers=self.providers)
            
            feature_maps = [[64, 64], [32, 32], [16, 16]]
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
            image_size = 512
            
            for k, f in enumerate(feature_maps):
                min_size_array = min_sizes[k]
                for i, j in product(range(f[0]), range(f[1])):
                    for min_size in min_size_array:
                        s_kx = min_size / image_size
                        s_ky = min_size / image_size
                        dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                        dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            self.anchors += [cx, cy, s_kx, s_ky]   

        # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = image.permute(1,2,0)
        
        # image = image - [104, 117, 123]
        mean = torch.tensor([104, 117, 123], dtype=torch.float32, device='cuda')
        image = torch.sub(image, mean)
        
        # image = image.transpose(2, 0, 1)
        # image = np.float32(image[np.newaxis,:,:,:])
        image = image.permute(2,0,1)
        image = torch.reshape(image, (1, 3, 512, 512))


        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device='cuda')
        
        # ort_inputs = {"input": image}        
        conf = torch.empty((1,10752,2), dtype=torch.float32, device='cuda').contiguous()
        landmarks = torch.empty((1,10752,10), dtype=torch.float32, device='cuda').contiguous()

        io_binding = self.resnet50_model.io_binding() 
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='conf', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,2), buffer_ptr=conf.data_ptr())
        io_binding.bind_output(name='landmarks', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,10), buffer_ptr=landmarks.data_ptr())
        
        # _, conf, landmarks = self.resnet_model.run(None, ort_inputs)        
        torch.cuda.synchronize('cuda')
        self.resnet50_model.run_with_iobinding(io_binding)        
        

        # conf = torch.from_numpy(conf)
        # scores = conf.squeeze(0).numpy()[:, 1]
        scores = torch.squeeze(conf)[:, 1]
        
        # landmarks = torch.from_numpy(landmarks)
        # landmarks = landmarks.to('cuda')        

        priors = torch.tensor(self.anchors).view(-1, 4)
        priors = priors.to('cuda')

        # pre = landmarks.squeeze(0) 
        pre = torch.squeeze(landmarks, 0)
        
        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)
        # landmarks = landmarks * scale1
        landmarks = torch.mul(landmarks, scale1)

        landmarks = landmarks.cpu().numpy()  

        # ignore low scores
        inds = torch.where(scores>score)[0]
        inds = inds.cpu().numpy()  
        scores = scores.cpu().numpy()  
        
        landmarks, scores = landmarks[inds], scores[inds]    

        # sort
        order = scores.argsort()[::-1]
        landmarks = landmarks[order][0]

        return np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])