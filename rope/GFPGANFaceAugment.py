import os
import sys
import cv2
import numpy as np
import timeit
import onnxruntime

class GFPGANFaceAugment():
    def __init__(self, use_gpu = False):
        self.ort_session = onnxruntime.InferenceSession( "./GFPGANv1.4.onnx", providers=["CUDAExecutionProvider"])
        self.net_input_name = self.ort_session.get_inputs()[0].name
        _,self.net_input_channels,self.net_input_height,self.net_input_width = self.ort_session.get_inputs()[0].shape
        self.net_output_count = len(self.ort_session.get_outputs())
        self.face_size = 512
        self.face_template = np.array([[192, 240], [319, 240], [257, 371]]) * (self.face_size / 512.0)
        self.upscale_factor = 2
        self.affine = False
        self.affine_matrix = None
    #@profile  
    def pre_process(self, img):
        img = cv2.resize(img, (512, 512))
        img = img / 255.0
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[:,:,0] = (img[:,:,0]-0.5)/0.5
        img[:,:,1] = (img[:,:,1]-0.5)/0.5
        img[:,:,2] = (img[:,:,2]-0.5)/0.5
        img = np.float32(img[np.newaxis,:,:,:])
        img = img.transpose(0, 3, 1, 2)
        return img
    def post_process(self, output, height, width):
        output = output.clip(-1,1)

        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (output * 255.0).round()
        if self.affine:
            inverse_affine = cv2.invertAffineTransform(self.affine_matrix)
            inverse_affine *= self.upscale_factor
            if self.upscale_factor > 1:
                extra_offset = 0.5 * self.upscale_factor
            else:
                extra_offset = 0
            inverse_affine[:, 2] += extra_offset
            inv_restored = cv2.warpAffine(output, inverse_affine, (width, height))
            mask = np.ones((self.face_size, self.face_size), dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (width, height))
            inv_mask_erosion = cv2.erode(
                inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            total_face_area = np.sum(inv_mask_erosion)
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            inv_soft_mask = inv_soft_mask[:, :, None]
            output = pasted_face
        else:
            inv_soft_mask = np.ones((height, width, 1), dtype=np.float32)
            output = cv2.resize(output, (width, height))
        return output, inv_soft_mask
    #@profile 
    def forward(self, img):
        height, width = img.shape[0], img.shape[1]
        img = self.pre_process(img)

        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        output = ort_outs[0][0]
        output, inv_soft_mask = self.post_process(output, height, width)
 
        output = output.astype(np.uint8)
        return output, inv_soft_mask
