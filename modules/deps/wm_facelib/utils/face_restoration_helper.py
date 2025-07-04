import cv2
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize
import comfy.model_management as model_management
from comfy.utils import tiled_scale # Import for upscaling

from wm_facelib.detection import init_detection_model
from wm_facelib.parsing import init_parsing_model
from wm_facelib.utils.misc import img2tensor, imwrite, is_gray, bgr2gray, adain_npy

dlib_model_url = {
    'face_detector': 'https://github.com/jnjaby/KEEP/releases/download/v1.0.0/mmod_human_face_detector-4cb19393.dat',
    'shape_predictor_5': 'https://github.com/jnjaby/KEEP/releases/download/v1.0.0/shape_predictor_5_face_landmarks-c4b1e980.dat'
}

class FaceRestoreHelper(object):
    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50',
                 save_ext='png',
                 template_3points=False,
                 pad_blur=False,
                 use_parse=False,
                 device=None,
                 model_rootpath=None):
        self.template_3points = template_3points
        self.upscale_factor = int(upscale_factor)
        self.crop_ratio = crop_ratio
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))
        self.det_model = det_model
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        self.use_parse = use_parse
        
        self.model_rootpath = model_rootpath
        if self.model_rootpath is None:
            print("WARNING (FaceRestoreHelper): model_rootpath is None. Model loading might rely on default relative paths.")

        if self.det_model == 'dlib':
            self.face_template = np.array([[686.77227723, 488.62376238], [586.77227723, 493.59405941],
                                           [337.91089109, 488.38613861], [
                                               437.95049505, 493.51485149],
                                           [513.58415842, 678.5049505]])
            self.face_template = self.face_template / (1024 // face_size)
        elif self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])
        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1: self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1: self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []

        if device is None:
            self.device = model_management.get_torch_device()
        else:
            self.device = device

        # init face detection model
        if self.det_model == 'dlib':
            dlib_face_det_path = os.path.join(self.model_rootpath if self.model_rootpath else 'weights/dlib', os.path.basename(dlib_model_url['face_detector']))
            dlib_sp5_path = os.path.join(self.model_rootpath if self.model_rootpath else 'weights/dlib', os.path.basename(dlib_model_url['shape_predictor_5']))
            
            if not os.path.exists(dlib_face_det_path):
                print(f"Dlib face detector model not found at {dlib_face_det_path}. Ensure it's downloaded.")
            if not os.path.exists(dlib_sp5_path):
                print(f"Dlib shape predictor model not found at {dlib_sp5_path}. Ensure it's downloaded.")

            self.face_detector, self.shape_predictor_5 = self.init_dlib_from_paths(
                dlib_face_det_path, dlib_sp5_path)
        else:
            self.face_detector = init_detection_model(
                self.det_model, half=False, device=self.device, model_rootpath=self.model_rootpath
            )

        # init face parsing model
        self.use_parse = use_parse
        if self.use_parse:
            self.face_parse = init_parsing_model(
                model_name='parsenet', device=self.device, model_rootpath=self.model_rootpath
            )
 
    def _run_upscaler(self, model, cv2_image):
        """Helper to run a comfy-native upscaler on a cv2 image."""
        if model is None:
            return cv2_image
        # Replicate cv2_to_comfy_image
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        img_np = img_rgb.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
        
        img_tensor_bchw = img_tensor.movedim(-1, -3)
        s = tiled_scale(img_tensor_bchw, lambda a: model.model(a), tile_x=512, tile_y=512, overlap=64, upscale_amount=model.scale)
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        
        # Replicate comfy_image_to_cv2
        img_np_out = s.cpu().numpy()
        img_np_out = (img_np_out.squeeze(0) * 255).astype(np.uint8)
        img_cv2_out = cv2.cvtColor(img_np_out, cv2.COLOR_RGB2BGR)
        return img_cv2_out

    #ToDo: remove dlib
    def init_dlib_from_paths(self, detection_model_path, landmark_model_path):
        """Initialize the dlib detectors and predictors from specified paths."""
        try:
            import dlib
        except ImportError:
            # print('Please install dlib by running: conda install -c conda-forge dlib or pip install dlib')
            raise
        
        if not os.path.exists(detection_model_path):
            raise FileNotFoundError(f"Dlib detection model not found: {detection_model_path}")
        if not os.path.exists(landmark_model_path):
            raise FileNotFoundError(f"Dlib landmark model not found: {landmark_model_path}")

        face_detector = dlib.cnn_face_detection_model_v1(detection_model_path)
        shape_predictor_5 = dlib.shape_predictor(landmark_model_path)
        return face_detector, shape_predictor_5


    def get_largest_face(self, det_faces, h, w):

        def get_location(val, length):
            if val < 0: return 0
            elif val > length: return length
            else: return val

        face_areas = []
        for det_face in det_faces:
            left = get_location(det_face[0], w)
            right = get_location(det_face[2], w)
            top = get_location(det_face[1], h)
            bottom = get_location(det_face[3], h)
            face_area = (right - left) * (bottom - top)
            face_areas.append(face_area)
        if not face_areas: return None, -1 # handle case with no valid faces
        largest_idx = face_areas.index(max(face_areas))
        return det_faces[largest_idx], largest_idx


    def get_center_face(self, det_faces, h=0, w=0, center=None):
        if not det_faces: return None, -1 # handle case with no faces
        if center is not None:
            center = np.array(center)
        else:
            center = np.array([w / 2, h / 2])
        center_dist = []
        for det_face in det_faces:
            face_center = np.array(
                [(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2])
            dist = np.linalg.norm(face_center - center)
            center_dist.append(dist)
        if not center_dist: return None, -1 # handle if somehow still empty
        center_idx = center_dist.index(min(center_dist))
        return det_faces[center_idx], center_idx

    def set_upscale_factor(self, upscale_factor):
        self.upscale_factor = upscale_factor

    def read_image(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        if img is None: raise ValueError("Image not loaded correctly.")
        if np.max(img) > 256: img = img / 65535 * 255
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4: img = img[:, :, 0:3]
        self.input_img = img
        self.is_gray = is_gray(img, threshold=10)
        if self.is_gray: print('Grayscale input: True')
        if min(self.input_img.shape[:2]) < 512:
            f = 512.0/min(self.input_img.shape[:2])
            self.input_img = cv2.resize(self.input_img, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR)

    def get_face_landmarks_5_dlib(self, only_keep_largest=False, scale_val=1): # renamed scale to scale_val
        det_faces = self.face_detector(self.input_img, scale_val) # dlib specific scaling
        if len(det_faces) == 0:
            return 0
        else:
            if only_keep_largest:
                face_areas = [(det.rect.right() - det.rect.left()) * (det.rect.bottom() - det.rect.top()) for det in det_faces]
                if not face_areas: return 0
                largest_idx = face_areas.index(max(face_areas))
                self.det_faces = [det_faces[largest_idx]]
            else:
                self.det_faces = det_faces
        if len(self.det_faces) == 0: return 0
        for face in self.det_faces:
            shape = self.shape_predictor_5(self.input_img, face.rect)
            landmark = np.array([[part.x, part.y] for part in shape.parts()])
            self.all_landmarks_5.append(landmark)
        return len(self.all_landmarks_5)

    def get_face_landmarks_5(self, only_center_face=False, resize=640, eye_dist_threshold=None, **kwargs):
        if self.det_model == 'dlib':
            return self.get_face_landmarks_5_dlib(only_center_face)

        img_for_detection = self.input_img
        h, w, _ = img_for_detection.shape
        if resize is not None and min(h, w) > resize:
            scale = resize / min(h, w)
            h_resized, w_resized = int(h * scale), int(w * scale)
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            img_for_detection = cv2.resize(self.input_img, (w_resized, h_resized), interpolation=interp)
            scale_for_bbox = h / h_resized
        else:
            scale_for_bbox = 1

        with torch.no_grad():
            bboxes = self.face_detector.detect_faces(img_for_detection, 0.97)
        
        if bboxes is None or bboxes.shape[0] == 0: return 0
        
        bboxes = bboxes * scale_for_bbox

        temp_landmarks = []
        temp_det_faces = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :5]
            landmark = bboxes[i, 5:].reshape((5, 2))

            if eye_dist_threshold is not None:
                eye_dist = np.linalg.norm(landmark[0] - landmark[1])
                if eye_dist < eye_dist_threshold:
                    continue

            temp_landmarks.append(landmark)
            temp_det_faces.append(bbox)
        
        if not temp_det_faces: return 0

        if only_center_face:
            h, w, _ = self.input_img.shape
            selected_face, selected_idx = self.get_center_face(temp_det_faces, h, w)
            if selected_idx != -1:
                self.det_faces = [selected_face]
                self.all_landmarks_5 = [temp_landmarks[selected_idx]]
            else:
                return 0
        else:
            self.det_faces = temp_det_faces
            self.all_landmarks_5 = temp_landmarks

        return len(self.all_landmarks_5)
        
    def align_warp_face(self, save_cropped_path=None, border_mode='constant'):
        # The pad_blur logic placed here, before alignment.
        if self.pad_blur:
            self.pad_input_imgs = []
            blur_ratio = 0.01  # This was a hardcoded parameter, kept it as is
            for landmarks in self.all_landmarks_5:
                eye_left, eye_right = landmarks[0, :], landmarks[1, :]
                eye_avg = (eye_left + eye_right) * 0.5
                mouth_avg = (landmarks[3, :] + landmarks[4, :]) * 0.5
                eye_to_eye = eye_right - eye_left
                eye_to_mouth = mouth_avg - eye_avg
                x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                x /= np.hypot(*x)
                rect_scale = 1.5
                x *= max(np.hypot(*eye_to_eye) * 2.0 * rect_scale, np.hypot(*eye_to_mouth) * 1.8 * rect_scale)
                y = np.flipud(x) * [-1, 1]
                c = eye_avg + eye_to_mouth * 0.1
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                qsize = np.hypot(*x) * 2
                border = max(int(np.rint(qsize * 0.1)), 3)
                pad = [int(np.floor(min(quad[:, i]))) for i in range(2)] + [int(np.ceil(max(quad[:, i]))) for i in range(2)]
                pad = [
                    max(-pad[0] + border, 1),
                    max(-pad[1] + border, 1),
                    max(pad[2] - self.input_img.shape[1] + border, 1),
                    max(pad[3] - self.input_img.shape[0] + border, 1)
                ]
                if max(pad) > 1:
                    pad_img = np.pad(self.input_img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                    h_pad, w_pad, _ = pad_img.shape
                    y_grid, x_grid = np.ogrid[:h_pad, :w_pad]
                    mask = np.maximum(
                        1.0 - np.minimum(np.float32(x_grid) / pad[0], np.float32(w_pad - 1 - x_grid) / pad[2]),
                        1.0 - np.minimum(np.float32(y_grid) / pad[1], np.float32(h_pad - 1 - y_grid) / pad[3])
                    )
                    blur = int(qsize * blur_ratio)
                    if blur % 2 == 0: blur += 1
                    blur_img = cv2.boxFilter(pad_img, 0, ksize=(blur, blur))
                    pad_img = pad_img.astype('float32')
                    pad_img += (blur_img - pad_img) * np.clip(mask[:, :, np.newaxis] * 3.0 + 1.0, 0.0, 1.0)
                    pad_img += (np.median(pad_img, axis=(0, 1)) - pad_img) * np.clip(mask[:, :, np.newaxis], 0.0, 1.0)
                    pad_img = np.clip(pad_img, 0, 255).astype(np.uint8)
                    self.pad_input_imgs.append(pad_img)
                else:
                    self.pad_input_imgs.append(np.copy(self.input_img))

        for idx, landmark in enumerate(self.all_landmarks_5):
            current_input_img = self.pad_input_imgs[idx] if self.pad_blur and idx < len(self.pad_input_imgs) else self.input_img

            affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
            if affine_matrix is None:
                self.cropped_faces.append(np.zeros((self.face_size[1], self.face_size[0], 3), dtype=np.uint8))
                self.affine_matrices.append(None)
                continue
            self.affine_matrices.append(affine_matrix)
            
            border_val = cv2.BORDER_CONSTANT
            if border_mode == 'reflect101': border_val = cv2.BORDER_REFLECT101
            elif border_mode == 'reflect': border_val = cv2.BORDER_REFLECT
            
            cropped_face = cv2.warpAffine(
                current_input_img, affine_matrix, self.face_size,
                borderMode=border_val, borderValue=(135, 133, 132))
            self.cropped_faces.append(cropped_face)
            if save_cropped_path is not None:
                path = os.path.splitext(save_cropped_path)[0]
                save_path = f'{path}_{idx:02d}.{self.save_ext}'
                imwrite(cropped_face, save_path)

    def get_inverse_affine(self, save_inverse_affine_path=None):
        self.inverse_affine_matrices = []
        for idx, affine_matrix in enumerate(self.affine_matrices):
            if affine_matrix is None:
                self.inverse_affine_matrices.append(None)
                continue
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
            inverse_affine *= self.upscale_factor
            self.inverse_affine_matrices.append(inverse_affine)
            if save_inverse_affine_path is not None:
                path, _ = os.path.splitext(save_inverse_affine_path)
                save_path = f'{path}_{idx:02d}.pth'
                torch.save(inverse_affine, save_path)

    def add_restored_face(self, restored_face, input_face=None):
        if self.is_gray:
            restored_face = bgr2gray(restored_face)
            if input_face is not None: restored_face = adain_npy(restored_face, input_face)
        self.restored_faces.append(restored_face)

    def paste_faces_to_input_image(self, save_path=None, upsample_img=None, draw_box=False, face_upsampler=None):
        if not hasattr(self, 'input_img') or not self.input_img.any():
            print("Error: input_img not set in FaceRestoreHelper. Call read_image() first.")
            return upsample_img if upsample_img is not None else cv2.cvtColor(np.zeros((100,100,3), dtype=np.uint8), cv2.COLOR_BGR2GRAY) if self.is_gray else np.zeros((100,100,3), dtype=np.uint8)

        h, w, _ = self.input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)

        if upsample_img is None:
            upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
        else:
            if upsample_img.shape[0] != h_up or upsample_img.shape[1] != w_up:
                 upsample_img = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        
        if len(self.restored_faces) != len(self.inverse_affine_matrices):
            print(f"Warning: Mismatch between restored_faces ({len(self.restored_faces)}) and inverse_affine_matrices ({len(self.inverse_affine_matrices)}). Problems may occur.")

        inv_mask_borders = []
        for idx, restored_face_orig in enumerate(self.restored_faces): 
            if idx >= len(self.inverse_affine_matrices) or self.inverse_affine_matrices[idx] is None:
                continue
            
            inverse_affine_matrix = self.inverse_affine_matrices[idx]
            current_restored_face = restored_face_orig.copy()

            if face_upsampler is not None:
                upscaled_face_patch = self._run_upscaler(face_upsampler, current_restored_face)
                target_w_for_warp, target_h_for_warp = self.face_size
                current_restored_face = cv2.resize(
                    upscaled_face_patch, (target_w_for_warp, target_h_for_warp),
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            mask_dimensions_for_warp = current_restored_face.shape[0:2]

            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 3 and len(current_restored_face.shape) == 2:
                current_restored_face = cv2.cvtColor(current_restored_face, cv2.COLOR_GRAY2BGR)

            inv_restored = cv2.warpAffine(current_restored_face, inverse_affine_matrix, (w_up, h_up))
            mask = np.ones(mask_dimensions_for_warp, dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine_matrix, (w_up, h_up))
            inv_mask_erosion = cv2.erode(
                inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
            
            if len(inv_restored.shape) == 2 and len(upsample_img.shape) == 3 and upsample_img.shape[2] == 3:
                pasted_face_content = cv2.cvtColor(inv_restored, cv2.COLOR_GRAY2BGR)
            else:
                pasted_face_content = inv_restored

            total_face_area = np.sum(inv_mask_erosion)
            if total_face_area == 0: total_face_area = 1

            if draw_box:
                h_fs, w_fs = mask_dimensions_for_warp
                mask_border = np.ones((h_fs, w_fs, 3), dtype=np.float32)
                border_thickness = int(1400 / np.sqrt(total_face_area))
                border_thickness = max(1, min(border_thickness, min(h_fs, w_fs) // 20))
                cv2.rectangle(mask_border, (border_thickness, border_thickness), (w_fs-border_thickness-1, h_fs-border_thickness-1), (0,0,0), -1)
                inv_mask_border_img = cv2.warpAffine(mask_border, inverse_affine_matrix, (w_up, h_up))
                inv_mask_borders.append(inv_mask_border_img)

            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = max(1, w_edge * 2)
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = max(1, w_edge * 2)
            if blur_size % 2 == 0: blur_size +=1
            
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size, blur_size), 0)
            if len(upsample_img.shape) == 2: upsample_img = upsample_img[:, :, None]
            inv_soft_mask = inv_soft_mask[:, :, None]

            if self.use_parse:
                face_input_for_parse = cv2.resize(current_restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                if len(face_input_for_parse.shape) == 2:
                    face_input_for_parse = cv2.cvtColor(face_input_for_parse, cv2.COLOR_GRAY2BGR)

                face_input_tensor = img2tensor(face_input_for_parse.astype('float32') / 255., bgr2rgb=True, float32=True)
                normalize(face_input_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_input_tensor = torch.unsqueeze(face_input_tensor, 0).to(self.device)
                
                with torch.no_grad():
                    out_parse = self.face_parse(face_input_tensor)[0]
                out_parse = out_parse.argmax(dim=1).squeeze().cpu().numpy()
                parse_mask_raw = np.zeros(out_parse.shape, dtype=np.float32)
                MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                for map_idx, color_val in enumerate(MASK_COLORMAP): parse_mask_raw[out_parse == map_idx] = color_val
                parse_mask_blurred = cv2.GaussianBlur(parse_mask_raw, (101, 101), 11)
                parse_mask_blurred = cv2.GaussianBlur(parse_mask_blurred, (101, 101), 11)
                thres = 10
                parse_mask_blurred[:thres, :] = 0; parse_mask_blurred[-thres:, :] = 0
                parse_mask_blurred[:, :thres] = 0; parse_mask_blurred[:, -thres:] = 0
                parse_mask_final = parse_mask_blurred / 255.
                
                parse_mask_final_resized = cv2.resize(parse_mask_final, (current_restored_face.shape[1], current_restored_face.shape[0]))
                
                inv_soft_parse_mask = cv2.warpAffine(parse_mask_final_resized, inverse_affine_matrix, (w_up, h_up), flags=cv2.INTER_LINEAR)[:, :, None]
                inv_soft_mask = inv_soft_parse_mask

            if len(upsample_img.shape) == 3 and len(inv_restored.shape) == 2:
                inv_restored_3c = cv2.cvtColor(inv_restored, cv2.COLOR_GRAY2BGR)
            else:
                inv_restored_3c = inv_restored
            
            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:
                alpha_channel = upsample_img[:, :, 3:]
                upsample_img_rgb = upsample_img[:, :, 0:3]
                upsample_img_rgb = inv_soft_mask * inv_restored_3c + (1 - inv_soft_mask) * upsample_img_rgb
                upsample_img = np.concatenate((upsample_img_rgb, alpha_channel), axis=2)
            else:
                upsample_img = inv_soft_mask * inv_restored_3c + (1 - inv_soft_mask) * upsample_img
        
        if np.issubdtype(upsample_img.dtype, np.floating):
             upsample_img = np.clip(upsample_img, 0, 255)
        
        output_final = upsample_img.round().astype(np.uint8)

        if draw_box:
            img_color_border = np.array([0,255,0], dtype=output_final.dtype)
            for inv_mask_border_item_img in inv_mask_borders:
                border_pixels_mask = np.any(inv_mask_border_item_img > 0.5, axis=2)
                if len(output_final.shape) == 3 and output_final.shape[2] == 4:
                     output_final[border_pixels_mask, 0:3] = img_color_border
                else:
                     output_final[border_pixels_mask] = img_color_border

        if save_path is not None:
            path_base = os.path.splitext(save_path)[0]
            save_path_final = f'{path_base}.{self.save_ext}'
            imwrite(output_final, save_path_final)
        return output_final

    def clean_all(self):
        self.all_landmarks_5 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
        self.det_faces = []
        self.pad_input_imgs = []

class FaceAligner(object):
    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50',
                 save_ext='png',
                 template_3points=False,
                 pad_blur=False,
                 use_parse=False,
                 device=None):
        self.template_3points = template_3points
        self.upscale_factor = int(upscale_factor)
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (self.crop_ratio[0] >= 1 and self.crop_ratio[1]
                >= 1), 'crop ration only supports >=1'
        self.face_size = (
            int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))
        self.det_model = det_model

        if self.det_model == 'dlib':
            # standard 5 landmarks for FFHQ faces with 1024 x 1024
            self.face_template = np.array([[686.77227723, 488.62376238], [586.77227723, 493.59405941],
                                           [337.91089109, 488.38613861], [
                                               437.95049505, 493.51485149],
                                           [513.58415842, 678.5049505]])
            self.face_template = self.face_template / (1024 // face_size)
        elif self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512
            # facexlib
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])

        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * \
                (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * \
                (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        if self.pad_blur is True:
            self.template_3points = False

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []

        if device is None:
            self.device = model_management.get_torch_device()
        else:
            self.device = device

    def set_image(self, img):
        self.input_img = img

    def align_pair_face(self, img_lq, img_gt, landmarks):
        img_lq = (img_lq[:, :, ::-1] * 255).round().astype(np.uint8)
        img_gt = (img_gt[:, :, ::-1] * 255).round().astype(np.uint8)

        self.set_image(img_gt)
        img_lq, img_gt = self.align_warp_face(img_lq, img_gt, landmarks)
        img_lq = img_lq[:, :, ::-1] / 255.0
        img_gt = img_gt[:, :, ::-1] / 255.0
        return img_lq, img_gt

    def align_single_face(self, img, landmarks, border_mode='constant'):
        """Align and warp faces with face template.
           Suppose input images are Numpy array, (h, w, c), BGR, uint8, [0, 255]
        """
        if border_mode == 'constant':
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == 'reflect101':
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == 'reflect':
            border_mode = cv2.BORDER_REFLECT

        img = (img[:, :, ::-1] * 255).round().astype(np.uint8)

        affine_matrix = cv2.estimateAffinePartial2D(
            landmarks, self.face_template, method=cv2.LMEDS)[0]
        img = cv2.warpAffine(
            img, affine_matrix, img.shape[0:2], borderMode=border_mode, borderValue=(135, 133, 132))  # gray
        img = img[:, :, ::-1] / 255.0
        return img

    def align_warp_face(self, img_lq, img_gt, landmarks, border_mode='constant'):
        """Align and warp faces with face template.
           Suppose input images are Numpy array, (h, w, c), BGR, uint8, [0, 255]
        """
        scale = img_gt.shape[0] / img_lq.shape[0]
        if border_mode == 'constant':
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == 'reflect101':
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == 'reflect':
            border_mode = cv2.BORDER_REFLECT

        affine_matrix = cv2.estimateAffinePartial2D(
            landmarks, self.face_template, method=cv2.LMEDS)[0]
        img_gt = cv2.warpAffine(
            img_gt, affine_matrix, img_gt.shape[0:2], borderMode=border_mode, borderValue=(135, 133, 132))  # gray

        affine_matrix = cv2.estimateAffinePartial2D(
            landmarks / scale, self.face_template / scale, method=cv2.LMEDS)[0]
        img_lq = cv2.warpAffine(
            img_lq, affine_matrix, img_lq.shape[0:2], borderMode=border_mode, borderValue=(135, 133, 132))  # gray

        return img_lq, img_gt

    def clean_all(self):
        self.all_landmarks_5 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
        self.det_faces = []
        self.pad_input_imgs = []