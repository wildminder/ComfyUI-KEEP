import torch
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from torchvision.transforms.functional import normalize
import os
from tqdm import tqdm # Import tqdm for console progress bars
from .utils import comfy_image_to_cv2, cv2_to_comfy_image
from comfy.utils import ProgressBar, tiled_scale

# Imports from vendored basicsr and facelib
try:
    from basicsr.utils.img_util import img2tensor, tensor2img
    from basicsr.utils.video_util import VideoReader, VideoWriter
    from facelib.utils.misc import is_gray
except ImportError as e:
    print(f"ImportError in keep_processor: {e}. Check vendored 'basicsr' and 'facelib'.")
    # Dummy just in case
    def img2tensor(img, bgr2rgb, float32): return None
    def tensor2img(tensor, rgb2bgr, min_max): return None
    def is_gray(img, threshold): return False
    class VideoReader:
        def __init__(self, path): self.path = path
        def __len__(self): return 0
        def get_frame(self): return None
        def get_fps(self): return 25
        def close(self): pass
    class VideoWriter:
        def __init__(self, path, h, w, fps): pass
        def write_frame(self, frame): pass
        def close(self): pass


def interpolate_sequence(sequence):
    interpolated_sequence = np.copy(sequence)
    missing_indices = np.isnan(sequence)
    if np.any(missing_indices):
        valid_indices = ~missing_indices
        x = np.arange(len(sequence))
        interpolated_sequence[missing_indices] = np.interp(
            x[missing_indices], x[valid_indices], sequence[valid_indices])
    return interpolated_sequence


class KEEPFaceProcessor:
    def __init__(self, model_pack):
        self.keep_net = model_pack.keep_net
        self.face_helper = model_pack.face_helper
        self.bg_upscale_model = model_pack.bg_upscale_model
        self.face_upscale_model = model_pack.face_upscale_model
        self.device = model_pack.device
        self.model_type_str = model_pack.model_type_str

    def _run_upscaler(self, model, cv2_image):
        """Helper to run a comfy-native upscaler on a cv2 image."""
        if model is None:
            return cv2_image

        img_tensor = cv2_to_comfy_image(cv2_image).to(self.device)
        img_tensor_bchw = img_tensor.movedim(-1, -3)
        s = tiled_scale(img_tensor_bchw, lambda a: model.model(a), tile_x=512, tile_y=512, overlap=64, upscale_amount=model.scale)
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return comfy_image_to_cv2(s)

    @torch.no_grad()
    def process_image(self, cv2_image_orig: np.ndarray, final_upscale_factor: float, has_aligned: bool,
                      only_center_face: bool, draw_box: bool):
        
        self.face_helper.upscale_factor = final_upscale_factor
        
        input_img_list = [cv2_image_orig.copy()]
        avg_landmarks_list = None

        if not has_aligned:
            raw_landmarks_list_np = []
            img_for_detection = input_img_list[0]
            self.face_helper.clean_all()
            self.face_helper.read_image(img_for_detection)
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5,
                only_keep_largest=True)

            if num_det_faces >= 1:
                raw_landmarks_list_np.append(self.face_helper.all_landmarks_5[0].reshape((10,)))
            else: return cv2_image_orig
            if not raw_landmarks_list_np: return cv2_image_orig
            raw_landmarks_np_array = np.array(raw_landmarks_list_np)
            for i_lm in range(raw_landmarks_np_array.shape[1]):
                raw_landmarks_np_array[:, i_lm] = interpolate_sequence(raw_landmarks_np_array[:, i_lm])
            avg_landmarks_np_array = raw_landmarks_np_array.reshape(raw_landmarks_np_array.shape[0], 5, 2)
            avg_landmarks_list = [lm for lm in avg_landmarks_np_array]
        
        cropped_face_t_list = []
        img_to_crop = input_img_list[0]
        if not has_aligned:
            if avg_landmarks_list is None or not avg_landmarks_list: return img_to_crop
            self.face_helper.clean_all()
            self.face_helper.read_image(img_to_crop)
            self.face_helper.all_landmarks_5 = [avg_landmarks_list[0]]
            self.face_helper.align_warp_face()
        else:
            img_resized = cv2.resize(img_to_crop, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.is_gray = is_gray(img_resized, threshold=10)
            self.face_helper.cropped_faces = [img_resized]
        if not self.face_helper.cropped_faces: return cv2_image_orig
        
        cropped_face_cv2 = self.face_helper.cropped_faces[0]
        cropped_face_t = img2tensor(cropped_face_cv2 / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t_list.append(cropped_face_t)
        cropped_faces_tensor = torch.stack(cropped_face_t_list, dim=0).unsqueeze(0).to(self.device)
        duplicated_input_frames = torch.cat([cropped_faces_tensor, cropped_faces_tensor], dim=1)
        net_output = self.keep_net(duplicated_input_frames, need_upscale=False)
        final_output_tensor = net_output[:, 0:1, ...].squeeze(0)
        restored_faces_cv2 = [tensor2img(x, rgb2bgr=True, min_max=(-1, 1)) for x in final_output_tensor]
        torch.cuda.empty_cache()

        img_for_pasting = input_img_list[0]
        self.face_helper.clean_all()
        if has_aligned:
            img_resized = cv2.resize(img_for_pasting, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.is_gray = is_gray(img_resized, threshold=10)
        else:
            if avg_landmarks_list is None or not avg_landmarks_list: return img_for_pasting
            self.face_helper.read_image(img_for_pasting)
            self.face_helper.all_landmarks_5 = [avg_landmarks_list[0]]
            self.face_helper.align_warp_face()

        if not restored_faces_cv2 or not restored_faces_cv2[0].any(): return cv2_image_orig
        
        self.face_helper.add_restored_face(restored_faces_cv2[0].astype('uint8'))

        if not has_aligned:
            bg_img_upscaled_by_model = self._run_upscaler(self.bg_upscale_model, img_for_pasting)

            h, w, _ = img_for_pasting.shape
            target_h, target_w = int(h * final_upscale_factor), int(w * final_upscale_factor)
            if bg_img_upscaled_by_model.shape[0] != target_h or bg_img_upscaled_by_model.shape[1] != target_w:
                bg_img_final = cv2.resize(bg_img_upscaled_by_model, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                bg_img_final = bg_img_upscaled_by_model

            self.face_helper.get_inverse_affine(None)
            final_restored_image = self.face_helper.paste_faces_to_input_image(
                upsample_img=bg_img_final, 
                draw_box=draw_box, 
                face_upsampler=self.face_upscale_model
            )
        else:
            final_restored_image = self.face_helper.restored_faces[0]
            if self.face_upscale_model:
                 final_restored_image = self._run_upscaler(self.face_upscale_model, final_restored_image)
            
            h, w, _ = final_restored_image.shape
            target_h, target_w = int(512 * final_upscale_factor), int(512 * final_upscale_factor)
            if h != target_h or w != target_w:
                 final_restored_image = cv2.resize(final_restored_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        if final_restored_image is None:
            return cv2_image_orig

        return final_restored_image

    @torch.no_grad()
    def process_video(self, video_path: str, output_video_path: str,
                      final_upscale_factor: float, has_aligned: bool, only_center_face: bool, draw_box: bool,
                      max_clip_length: int = 20, target_fps: float = None):

        vidreader = VideoReader(video_path)
        original_fps = vidreader.get_fps()
        num_input_frames = len(vidreader)

        pbar_comfy = ProgressBar(num_input_frames * 5)

        input_frames_bgr = []
        for _ in tqdm(range(num_input_frames), desc="Reading video frames"):
            img = vidreader.get_frame()
            if img is None: break
            input_frames_bgr.append(img)
            pbar_comfy.update(1)
        vidreader.close()

        if not input_frames_bgr: return None

        avg_landmarks_sequence = None
        if not has_aligned:
            raw_landmarks_list_video = []
            for i in tqdm(range(num_input_frames), desc="Detecting face landmarks"):
                img = input_frames_bgr[i]
                self.face_helper.clean_all()
                self.face_helper.read_image(img)
                num_det_faces = self.face_helper.get_face_landmarks_5(
                    only_center_face=only_center_face, resize=640, eye_dist_threshold=5,
                    only_keep_largest=True)
                if num_det_faces >= 1:
                    raw_landmarks_list_video.append(self.face_helper.all_landmarks_5[0].reshape((10,)))
                else:
                    raw_landmarks_list_video.append(np.array([np.nan] * 10))
                pbar_comfy.update(1)
            raw_landmarks_np_video = np.array(raw_landmarks_list_video)
            if np.all(np.isnan(raw_landmarks_np_video)): return None
            for i_lm_coord in range(raw_landmarks_np_video.shape[1]):
                raw_landmarks_np_video[:, i_lm_coord] = interpolate_sequence(raw_landmarks_np_video[:, i_lm_coord])
            avg_landmarks_sequence = gaussian_filter1d(raw_landmarks_np_video, sigma=5, axis=0).reshape(num_input_frames, 5, 2)
        else:
            pbar_comfy.update(num_input_frames)

        all_cropped_face_tensors = []
        for i in tqdm(range(num_input_frames), desc="Cropping and aligning faces"):
            img_bgr = input_frames_bgr[i]
            self.face_helper.clean_all()
            self.face_helper.read_image(img_bgr)
            if not has_aligned:
                if avg_landmarks_sequence is None or i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                    all_cropped_face_tensors.append(torch.zeros(3, 512, 512, dtype=torch.float32, device=self.device) * 0.5)
                    continue
                self.face_helper.all_landmarks_5 = [avg_landmarks_sequence[i]]
                self.face_helper.align_warp_face()
            else:
                img_resized = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
                self.face_helper.is_gray = is_gray(img_resized, threshold=10)
                self.face_helper.cropped_faces = [img_resized]
            if not self.face_helper.cropped_faces:
                all_cropped_face_tensors.append(torch.zeros(3, 512, 512, dtype=torch.float32, device=self.device) * 0.5)
                continue
            cropped_face_cv2 = self.face_helper.cropped_faces[0]
            cropped_face_t = img2tensor(cropped_face_cv2 / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            all_cropped_face_tensors.append(cropped_face_t)
            pbar_comfy.update(1)

        if not all_cropped_face_tensors: return None
            
        batched_cropped_faces = torch.stack(all_cropped_face_tensors, dim=0).unsqueeze(0).to(self.device)
        temp_restored_face_tensors = []
        for start_idx in tqdm(range(0, num_input_frames, max_clip_length), desc="Restoring faces with KEEP"):
            end_idx = min(start_idx + max_clip_length, num_input_frames)
            current_clip_faces = batched_cropped_faces[:, start_idx:end_idx, ...]
            if current_clip_faces.shape[1] == 1:
                 current_clip_faces_for_net = torch.cat([current_clip_faces, current_clip_faces], dim=1)
                 clip_output_tensor = self.keep_net(current_clip_faces_for_net, need_upscale=False)
                 temp_restored_face_tensors.append(clip_output_tensor[:, 0:1, ...])
            elif current_clip_faces.shape[1] > 1 :
                 clip_output_tensor = self.keep_net(current_clip_faces, need_upscale=False)
                 temp_restored_face_tensors.append(clip_output_tensor)
            pbar_comfy.update(end_idx - start_idx)
        
        if not temp_restored_face_tensors: return None
        all_restored_face_tensors = torch.cat(temp_restored_face_tensors, dim=1).squeeze(0)
        all_restored_faces_cv2 = [tensor2img(t, rgb2bgr=True, min_max=(-1, 1)) for t in all_restored_face_tensors]
        del batched_cropped_faces, all_restored_face_tensors, temp_restored_face_tensors
        torch.cuda.empty_cache()
        
        final_fps = target_fps if target_fps is not None else original_fps
        h_orig, w_orig, _ = input_frames_bgr[0].shape
        output_video_height, output_video_width = int(h_orig * final_upscale_factor), int(w_orig * final_upscale_factor)

        vidwriter = VideoWriter(output_video_path, output_video_height, output_video_width, final_fps)
        
        for i in tqdm(range(num_input_frames), desc="Pasting faces and writing video"):
            original_frame_bgr = input_frames_bgr[i]
            if i >= len(all_restored_faces_cv2):
                final_frame_to_write = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                if final_frame_to_write.shape[0] != output_video_height or final_frame_to_write.shape[1] != output_video_width:
                    final_frame_to_write = cv2.resize(final_frame_to_write, (output_video_width, output_video_height))
                vidwriter.write_frame(final_frame_to_write)
                continue

            restored_face_patch_cv2 = all_restored_faces_cv2[i]
            self.face_helper.clean_all()
            self.face_helper.read_image(original_frame_bgr)
            self.face_helper.add_restored_face(restored_face_patch_cv2.astype('uint8'))

            if not has_aligned:
                if avg_landmarks_sequence is None or i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                    final_frame_to_write = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                    if final_frame_to_write.shape[0] != output_video_height or final_frame_to_write.shape[1] != output_video_width:
                         final_frame_to_write = cv2.resize(final_frame_to_write, (output_video_width, output_video_height))
                    vidwriter.write_frame(final_frame_to_write)
                    continue
                
                self.face_helper.upscale_factor = final_upscale_factor
                self.face_helper.all_landmarks_5 = [avg_landmarks_sequence[i]]
                self.face_helper.align_warp_face()
                
                bg_img_upscaled_by_model = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                if bg_img_upscaled_by_model.shape[0] != output_video_height or bg_img_upscaled_by_model.shape[1] != output_video_width:
                    bg_img_final = cv2.resize(bg_img_upscaled_by_model, (output_video_width, output_video_height), interpolation=cv2.INTER_LANCZOS4)
                else:
                    bg_img_final = bg_img_upscaled_by_model
                
                self.face_helper.get_inverse_affine(None)
                pasted_frame = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img_final, draw_box=draw_box, face_upsampler=self.face_upscale_model)
            else:
                pasted_frame = self.face_helper.restored_faces[0]
                if self.face_upscale_model:
                    pasted_frame = self._run_upscaler(self.face_upscale_model, pasted_frame)
                if len(pasted_frame.shape) == 2: pasted_frame = cv2.cvtColor(pasted_frame, cv2.COLOR_GRAY2BGR)
            
            if pasted_frame.shape[0] != output_video_height or pasted_frame.shape[1] != output_video_width:
                pasted_frame = cv2.resize(pasted_frame, (output_video_width, output_video_height), interpolation=cv2.INTER_LANCZOS4)

            vidwriter.write_frame(pasted_frame)
            pbar_comfy.update(1)
        
        vidwriter.close()
        return output_video_path

    @torch.no_grad()
    def process_image_sequence(self,
                               image_sequence_tensor: torch.Tensor,
                               final_upscale_factor: float,
                               has_aligned_frames: bool,
                               only_center_face: bool,
                               draw_box: bool,
                               max_clip_length: int = 20):

        num_input_frames = image_sequence_tensor.shape[0]
        if num_input_frames == 0:
            return image_sequence_tensor

        pbar_comfy = ProgressBar(num_input_frames * 4)

        input_frames_bgr = []
        for i in range(num_input_frames):
            frame_tensor = image_sequence_tensor[i].unsqueeze(0) 
            input_frames_bgr.append(comfy_image_to_cv2(frame_tensor))
            pbar_comfy.update(1)
        
        avg_landmarks_sequence = None
        if not has_aligned_frames:
            raw_landmarks_list_video = []
            for i in tqdm(range(num_input_frames), desc="Detecting face landmarks"):
                img = input_frames_bgr[i]
                self.face_helper.clean_all()
                self.face_helper.read_image(img)
                num_det_faces = self.face_helper.get_face_landmarks_5(
                    only_center_face=only_center_face, resize=640, eye_dist_threshold=5,
                    only_keep_largest=True)
                if num_det_faces >= 1:
                    raw_landmarks_list_video.append(self.face_helper.all_landmarks_5[0].reshape((10,)))
                else:
                    raw_landmarks_list_video.append(np.array([np.nan] * 10))
            raw_landmarks_np_video = np.array(raw_landmarks_list_video)
            if np.all(np.isnan(raw_landmarks_np_video)):
                return image_sequence_tensor 
            for i_lm_coord in range(raw_landmarks_np_video.shape[1]):
                raw_landmarks_np_video[:, i_lm_coord] = interpolate_sequence(raw_landmarks_np_video[:, i_lm_coord])
            avg_landmarks_sequence = gaussian_filter1d(raw_landmarks_np_video, sigma=5, axis=0).reshape(num_input_frames, 5, 2)
            pbar_comfy.update(num_input_frames)
        else:
             pbar_comfy.update(num_input_frames)

        all_cropped_face_tensors = []
        for i in tqdm(range(num_input_frames), desc="Cropping and aligning faces"):
            img_bgr = input_frames_bgr[i]
            self.face_helper.clean_all()
            self.face_helper.read_image(img_bgr)
            if not has_aligned_frames:
                if avg_landmarks_sequence is None or i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                    all_cropped_face_tensors.append(torch.zeros(3, 512, 512, dtype=torch.float32, device=self.device) * 0.5)
                    continue
                self.face_helper.all_landmarks_5 = [avg_landmarks_sequence[i]]
                self.face_helper.align_warp_face()
            else:
                img_resized = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
                self.face_helper.is_gray = is_gray(img_resized, threshold=10)
                self.face_helper.cropped_faces = [img_resized]
            if not self.face_helper.cropped_faces:
                all_cropped_face_tensors.append(torch.zeros(3, 512, 512, dtype=torch.float32, device=self.device) * 0.5)
                continue
            cropped_face_cv2 = self.face_helper.cropped_faces[0]
            cropped_face_t = img2tensor(cropped_face_cv2 / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            all_cropped_face_tensors.append(cropped_face_t)

        if not all_cropped_face_tensors:
            return image_sequence_tensor

        batched_cropped_faces = torch.stack(all_cropped_face_tensors, dim=0).unsqueeze(0).to(self.device)
        temp_restored_face_tensors = []
        for start_idx in tqdm(range(0, num_input_frames, max_clip_length), desc="Restoring faces with KEEP"):
            end_idx = min(start_idx + max_clip_length, num_input_frames)
            current_clip_faces = batched_cropped_faces[:, start_idx:end_idx, ...]
            if current_clip_faces.shape[1] == 0: continue
            if current_clip_faces.shape[1] == 1:
                 current_clip_faces_for_net = torch.cat([current_clip_faces, current_clip_faces], dim=1)
                 clip_output_tensor = self.keep_net(current_clip_faces_for_net, need_upscale=False)
                 temp_restored_face_tensors.append(clip_output_tensor[:, 0:1, ...])
            else:
                 clip_output_tensor = self.keep_net(current_clip_faces, need_upscale=False)
                 temp_restored_face_tensors.append(clip_output_tensor)
            pbar_comfy.update(end_idx - start_idx)

        if not temp_restored_face_tensors:
            return image_sequence_tensor
        all_restored_face_tensors = torch.cat(temp_restored_face_tensors, dim=1).squeeze(0)
        all_restored_faces_cv2 = [tensor2img(t, rgb2bgr=True, min_max=(-1, 1)) for t in all_restored_face_tensors]
        del batched_cropped_faces, all_restored_face_tensors, temp_restored_face_tensors
        torch.cuda.empty_cache()

        output_frames_cv2 = []
        for i in tqdm(range(num_input_frames), desc="Pasting faces and finalizing frames"):
            original_frame_bgr = input_frames_bgr[i]
            if i >= len(all_restored_faces_cv2):
                final_pasted_frame = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                output_frames_cv2.append(final_pasted_frame)
                continue
            restored_face_patch_cv2 = all_restored_faces_cv2[i]
            self.face_helper.clean_all()
            self.face_helper.read_image(original_frame_bgr)
            self.face_helper.add_restored_face(restored_face_patch_cv2.astype('uint8'))

            if not has_aligned_frames:
                if avg_landmarks_sequence is None or i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                    final_pasted_frame = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                    output_frames_cv2.append(final_pasted_frame)
                    continue
                self.face_helper.all_landmarks_5 = [avg_landmarks_sequence[i]]
                self.face_helper.align_warp_face() 
                
                bg_img_upscaled_by_model = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                h, w, _ = original_frame_bgr.shape
                target_h, target_w = int(h * final_upscale_factor), int(w * final_upscale_factor)
                if bg_img_upscaled_by_model.shape[0] != target_h or bg_img_upscaled_by_model.shape[1] != target_w:
                    bg_img_final = cv2.resize(bg_img_upscaled_by_model, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    bg_img_final = bg_img_upscaled_by_model
                
                self.face_helper.upscale_factor = final_upscale_factor
                self.face_helper.get_inverse_affine(None)
                final_pasted_frame = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img_final,
                    draw_box=draw_box,
                    face_upsampler=self.face_upscale_model
                )
            else:
                restored_face_with_model = self.face_helper.restored_faces[0]
                if self.face_upscale_model:
                    restored_face_with_model = self._run_upscaler(self.face_upscale_model, restored_face_with_model)
                
                h_restored, w_restored, _ = restored_face_with_model.shape
                target_h, target_w = int(512 * final_upscale_factor), int(512 * final_upscale_factor)
                if h_restored != target_h or w_restored != target_w:
                    final_pasted_frame = cv2.resize(restored_face_with_model, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    final_pasted_frame = restored_face_with_model

                if len(final_pasted_frame.shape) == 2: 
                    final_pasted_frame = cv2.cvtColor(final_pasted_frame, cv2.COLOR_GRAY2BGR)

            output_frames_cv2.append(final_pasted_frame)
            pbar_comfy.update(1)
            
        output_tensors = [cv2_to_comfy_image(frame) for frame in output_frames_cv2]
        if not output_tensors:
             return image_sequence_tensor
        final_batch_tensor = torch.cat(output_tensors, dim=0)
        return final_batch_tensor