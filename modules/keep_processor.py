import torch
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from torchvision.transforms.functional import normalize
import os
from tqdm import tqdm
from .utils import comfy_image_to_cv2, cv2_to_comfy_image
from comfy.utils import ProgressBar, tiled_scale

try:
    from wm_basicsr.utils.img_util import img2tensor, tensor2img
    from wm_basicsr.utils.video_util import VideoReader, VideoWriter
    from wm_facelib.utils.misc import is_gray
except ImportError as e:
    print(f"ImportError in keep_processor: {e}. Check vendored 'wm_basicsr' and 'wm_facelib'.")
    # Dummy fallbacks
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
        interpolated_sequence[missing_indices] = np.interp(x[missing_indices], x[valid_indices], sequence[valid_indices])
    return interpolated_sequence

def track_faces(all_frames_landmarks, distance_threshold=75.0):
    """
    Tracks faces across frames using landmark proximity and the Hungarian algorithm.
    This version is robust to frames with zero detections.

    Args:
        all_frames_landmarks (list[list[np.array]]): A list where each element is a list of landmark arrays for that frame.
        distance_threshold (float): Maximum distance between landmark centroids for a match.

    Returns:
        dict: A dictionary where keys are track_ids and values are lists of landmark arrays for each frame.
              Missing frames are represented by np.nan.
    """
    tracks = {}
    next_track_id = 0
    num_frames = len(all_frames_landmarks)
    
    # Initialize with the first frame's detections
    if all_frames_landmarks and all_frames_landmarks[0]:
        for landmark in all_frames_landmarks[0]:
            tracks[next_track_id] = [landmark]
            next_track_id += 1

    # Iterate through the rest of the frames
    for i in range(1, num_frames):
        # Ensure all existing tracks have an entry for the previous frame (i-1)
        for track_id in tracks:
            if len(tracks[track_id]) < i:
                tracks[track_id].append(np.full((5, 2), np.nan))

        # Get last known landmarks from active tracks at frame i-1
        prev_landmarks = []
        active_track_ids = []
        for track_id, track_data in tracks.items():
            if len(track_data) == i and not np.all(np.isnan(track_data[-1])):
                prev_landmarks.append(track_data[-1])
                active_track_ids.append(track_id)

        current_landmarks = all_frames_landmarks[i]

        matched_current_indices = set()
        
        if prev_landmarks and current_landmarks:
            cost_matrix = np.full((len(prev_landmarks), len(current_landmarks)), np.inf)
            for r, prev_lm in enumerate(prev_landmarks):
                for c, cur_lm in enumerate(current_landmarks):
                    dist = np.linalg.norm(prev_lm.mean(axis=0) - cur_lm.mean(axis=0))
                    if dist < distance_threshold:
                        cost_matrix[r, c] = dist
            
            if not np.all(np.isinf(cost_matrix)):
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] != np.inf:
                        track_id = active_track_ids[r]
                        tracks[track_id].append(current_landmarks[c])
                        matched_current_indices.add(c)

        for track_id in active_track_ids:
            if len(tracks[track_id]) == i:
                tracks[track_id].append(np.full((5, 2), np.nan))

        unmatched_current_indices = set(range(len(current_landmarks))) - matched_current_indices
        for c_idx in unmatched_current_indices:
            tracks[next_track_id] = [np.full((5, 2), np.nan)] * i
            tracks[next_track_id].append(current_landmarks[c_idx])
            next_track_id += 1

    for track_id in tracks:
        while len(tracks[track_id]) < num_frames:
            tracks[track_id].append(np.full((5, 2), np.nan))

    return tracks

class KEEPFaceProcessor:
    def __init__(self, model_pack):
        self.keep_net = model_pack.keep_net
        self.face_helper = model_pack.face_helper
        self.bg_upscale_model = model_pack.bg_upscale_model
        self.face_upscale_model = model_pack.face_upscale_model
        self.device = model_pack.device
        self.model_type_str = model_pack.model_type_str

    def _run_upscaler(self, model, cv2_image):
        if model is None: return cv2_image
        img_tensor = cv2_to_comfy_image(cv2_image).to(self.device)
        img_tensor_bchw = img_tensor.movedim(-1, -3)
        s = tiled_scale(img_tensor_bchw, lambda a: model.model(a), tile_x=512, tile_y=512, overlap=64, upscale_amount=model.scale)
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return comfy_image_to_cv2(s)

    @torch.no_grad()
    def process_image(self, cv2_image_orig: np.ndarray, final_upscale_factor: float, has_aligned: bool,
                      only_center_face: bool, draw_box: bool):
        self.face_helper.upscale_factor = final_upscale_factor
        
        bg_img_upscaled = self._run_upscaler(self.bg_upscale_model, cv2_image_orig)
        h, w, _ = cv2_image_orig.shape
        target_h, target_w = int(h * final_upscale_factor), int(w * final_upscale_factor)
        bg_img_final = cv2.resize(bg_img_upscaled, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        if not has_aligned:
            self.face_helper.clean_all()
            self.face_helper.read_image(cv2_image_orig)
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
            if num_det_faces == 0:
                # print("No face detected. Returning upscaled background.")
                return bg_img_final
        
        cropped_face_t_list = []
        if not has_aligned:
            self.face_helper.align_warp_face()
            if not self.face_helper.cropped_faces: return bg_img_final
            for cropped_face_cv2 in self.face_helper.cropped_faces:
                cropped_face_t = img2tensor(cropped_face_cv2 / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t_list.append(cropped_face_t)
        else:
            img_resized = cv2.resize(cv2_image_orig, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.is_gray = is_gray(img_resized, threshold=10)
            self.face_helper.cropped_faces = [img_resized]
            cropped_face_t = img2tensor(img_resized / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t_list.append(cropped_face_t)
            
        if not cropped_face_t_list: return bg_img_final

        cropped_faces_tensor = torch.stack(cropped_face_t_list, dim=0).unsqueeze(0).to(self.device)
        
        if cropped_faces_tensor.shape[1] == 1:
            net_output = self.keep_net(torch.cat([cropped_faces_tensor, cropped_faces_tensor], dim=1), need_upscale=False)
            final_output_tensor = net_output[:, 0:1, ...].squeeze(0)
        else:
            net_output = self.keep_net(cropped_faces_tensor, need_upscale=False)
            final_output_tensor = net_output.squeeze(0)

        restored_faces_cv2 = [tensor2img(x, rgb2bgr=True, min_max=(-1, 1)) for x in final_output_tensor]
        torch.cuda.empty_cache()

        self.face_helper.restored_faces = [face.astype('uint8') for face in restored_faces_cv2]
        
        if not has_aligned:
            self.face_helper.get_inverse_affine(None)
            final_restored_image = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img_final, draw_box=draw_box, face_upsampler=self.face_upscale_model)
        else:
            final_restored_image = self.face_helper.restored_faces[0]
            if self.face_upscale_model: final_restored_image = self._run_upscaler(self.face_upscale_model, final_restored_image)
            target_h, target_w = int(512 * final_upscale_factor), int(512 * final_upscale_factor)
            if final_restored_image.shape[0] != target_h or final_restored_image.shape[1] != target_w: final_restored_image = cv2.resize(final_restored_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        return final_restored_image if final_restored_image is not None else bg_img_final

    @torch.no_grad()
    def process_image_sequence(self, image_sequence_tensor: torch.Tensor, final_upscale_factor: float, has_aligned_frames: bool, only_center_face: bool, draw_box: bool, max_clip_length: int = 20):
        num_input_frames = image_sequence_tensor.shape[0]
        if num_input_frames == 0: return image_sequence_tensor
        pbar_comfy = ProgressBar(num_input_frames * 4)
        input_frames_bgr = [comfy_image_to_cv2(image_sequence_tensor[i].unsqueeze(0)) for i in range(num_input_frames)]
        
        all_smoothed_landmarks = {}
        if not has_aligned_frames:
            # Step 1: Detect faces in all frames
            all_frames_landmarks_raw = []
            for i in tqdm(range(num_input_frames), desc="Detecting face landmarks"):
                self.face_helper.clean_all()
                self.face_helper.read_image(input_frames_bgr[i])
                num_detected = self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
                # Logging number of detected faces ***
                # print(f"Frame {i}: Detected {num_detected} faces.")
                all_frames_landmarks_raw.append(list(self.face_helper.all_landmarks_5))
            pbar_comfy.update(num_input_frames)

            if only_center_face:
                # Simple path for single face: no tracking needed, just smooth the sequence of (at most) one face
                single_face_sequence = [frame_lms[0] if frame_lms else np.full((5, 2), np.nan) for frame_lms in all_frames_landmarks_raw]
                landmark_seq_np = np.array([lm.reshape((10,)) for lm in single_face_sequence])
                for i_lm_coord in range(10):
                    landmark_seq_np[:, i_lm_coord] = interpolate_sequence(landmark_seq_np[:, i_lm_coord])
                all_smoothed_landmarks[0] = gaussian_filter1d(landmark_seq_np, sigma=2, axis=0).reshape(num_input_frames, 5, 2)
            else:
                if any(all_frames_landmarks_raw):
                    tracked_landmarks = track_faces(all_frames_landmarks_raw)
                    if tracked_landmarks:
                        for track_id, landmarks in tracked_landmarks.items():
                            landmark_seq_np = np.array([lm.reshape((10,)) if not np.all(np.isnan(lm)) else np.full((10,), np.nan) for lm in landmarks])
                            for i_lm_coord in range(10):
                                landmark_seq_np[:, i_lm_coord] = interpolate_sequence(landmark_seq_np[:, i_lm_coord])
                            all_smoothed_landmarks[track_id] = gaussian_filter1d(landmark_seq_np, sigma=2, axis=0).reshape(num_input_frames, 5, 2)
            pbar_comfy.update(num_input_frames)
        else:
             pbar_comfy.update(num_input_frames * 2)

        # Step 2 & 3: Crop and Restore all faces
        all_cropped_face_tensors, all_affine_matrices, face_counts_per_frame = [], [], []
        for i in tqdm(range(num_input_frames), desc="Cropping and aligning faces"):
            frame_cropped_faces, frame_affine_matrices = [], []
            if not has_aligned_frames:
                active_landmarks = [seq[i] for seq in all_smoothed_landmarks.values() if not np.isnan(seq[i]).any()]
                if active_landmarks:
                    self.face_helper.clean_all()
                    self.face_helper.read_image(input_frames_bgr[i])
                    self.face_helper.all_landmarks_5 = active_landmarks
                    self.face_helper.align_warp_face()
                    frame_cropped_faces.extend(self.face_helper.cropped_faces)
                    frame_affine_matrices.extend(self.face_helper.affine_matrices)
            else:
                frame_cropped_faces.append(cv2.resize(input_frames_bgr[i], (512, 512), interpolation=cv2.INTER_LINEAR))
            
            face_counts_per_frame.append(len(frame_cropped_faces))
            all_cropped_face_tensors.extend(frame_cropped_faces)
            all_affine_matrices.extend(frame_affine_matrices)

        all_restored_faces_cv2 = []
        if all_cropped_face_tensors:
            flat_face_tensors = [img2tensor(face / 255., bgr2rgb=True, float32=True) for face in all_cropped_face_tensors]
            for t in flat_face_tensors: normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            batched_cropped_faces = torch.stack(flat_face_tensors, dim=0).unsqueeze(0).to(self.device)
            temp_restored_tensors = []
            num_total_faces = batched_cropped_faces.shape[1]
            for start_idx in tqdm(range(0, num_total_faces, max_clip_length), desc="Restoring faces with KEEP"):
                end_idx = min(start_idx + max_clip_length, num_total_faces)
                current_clip = batched_cropped_faces[:, start_idx:end_idx, ...]
                if current_clip.shape[1] == 1:
                    current_clip = torch.cat([current_clip, current_clip], dim=1)
                    temp_restored_tensors.append(self.keep_net(current_clip, need_upscale=False)[:, 0:1, ...])
                elif current_clip.shape[1] > 1:
                    temp_restored_tensors.append(self.keep_net(current_clip, need_upscale=False))
            if temp_restored_tensors:
                all_restored_tensors = torch.cat(temp_restored_tensors, dim=1).squeeze(0)
                all_restored_faces_cv2 = [tensor2img(t, rgb2bgr=True, min_max=(-1, 1)) for t in all_restored_tensors]
            del batched_cropped_faces, temp_restored_tensors
            torch.cuda.empty_cache()
        pbar_comfy.update(num_input_frames)
        
        # Step 4: Paste faces back
        output_frames_cv2 = []
        restored_face_idx_counter, affine_matrix_idx_counter = 0, 0
        for i in tqdm(range(num_input_frames), desc="Pasting faces and finalizing frames"):
            original_frame_bgr = input_frames_bgr[i]
            bg_img_upscaled = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
            h, w, _ = original_frame_bgr.shape
            target_h, target_w = int(h * final_upscale_factor), int(w * final_upscale_factor)
            bg_img_final = cv2.resize(bg_img_upscaled, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            num_faces_this_frame = face_counts_per_frame[i]
            
            if num_faces_this_frame == 0 or has_aligned_frames:
                output_frames_cv2.append(bg_img_final) # a little simplified, aligned case could be handled better
                continue

            self.face_helper.restored_faces = [face.astype('uint8') for face in all_restored_faces_cv2[restored_face_idx_counter:restored_face_idx_counter + num_faces_this_frame]]
            self.face_helper.affine_matrices = all_affine_matrices[affine_matrix_idx_counter:affine_matrix_idx_counter + num_faces_this_frame]
            self.face_helper.upscale_factor = final_upscale_factor
            self.face_helper.get_inverse_affine(None)
            
            final_pasted_frame = self.face_helper.paste_faces_to_input_image(
                upsample_img=bg_img_final, draw_box=draw_box, face_upsampler=self.face_upscale_model)
            output_frames_cv2.append(final_pasted_frame)
            
            restored_face_idx_counter += num_faces_this_frame
            affine_matrix_idx_counter += num_faces_this_frame
            pbar_comfy.update(1)

        output_tensors = [cv2_to_comfy_image(frame) for frame in output_frames_cv2]
        return torch.cat(output_tensors, dim=0) if output_tensors else image_sequence_tensor