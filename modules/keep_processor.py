import torch
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from torchvision.transforms.functional import normalize
import os
from tqdm import tqdm
from .utils import comfy_image_to_cv2, cv2_to_comfy_image

# Imports from vendored basicsr and facelib
try:
    from basicsr.utils.img_util import img2tensor, tensor2img
    from basicsr.utils.video_util import VideoReader, VideoWriter # <-- NEW
    from facelib.utils.misc import is_gray
except ImportError as e:
    print(f"ImportError in keep_processor: {e}. Check vendored 'basicsr' and 'facelib'.")
    # Dummy just in case, dunno if needed
    def img2tensor(img, bgr2rgb, float32): return None
    def tensor2img(tensor, rgb2bgr, min_max): return None
    def is_gray(img, threshold): return False
    class VideoReader:
        def __init__(self, path): self.path = path
        def get_frame(self): return None
        def get_fps(self): return 25
        def close(self): pass
        def __len__(self): return 0
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
        self.bg_upsampler = model_pack.bg_upsampler
        self.face_upsampler = model_pack.face_upsampler
        self.device = model_pack.device
        self.model_type_str = model_pack.model_type_str

    @torch.no_grad()
    def process_image(self, cv2_image_orig: np.ndarray, has_aligned: bool,
                      only_center_face: bool, draw_box: bool, upscale_factor: int = 1,
                      max_length_override: int = 20): # max_length mostly for consistency

        self.face_helper.upscale_factor = upscale_factor

        input_img_list = [cv2_image_orig.copy()]
        avg_landmarks_list = None

        if not has_aligned:
            # print('Detecting keypoints and preparing alignment...')
            # Use a list to accumulate landmarks from potentially multiple faces per image (though we only use one)
            raw_landmarks_list_np = []
            
            # For single image, this loop runs once
            img_for_detection = input_img_list[0]
            self.face_helper.clean_all()
            self.face_helper.read_image(img_for_detection)
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5,
                only_keep_largest=True)

            if num_det_faces >= 1:
                # Take the first detected face's landmarks
                raw_landmarks_list_np.append(self.face_helper.all_landmarks_5[0].reshape((10,)))
            else:
                print("Warning: No face detected in the input image. Returning original.")
                return cv2_image_orig

            # Should not happen if num_det_faces >=1
            if not raw_landmarks_list_np:
                print("Warning: Landmark detection failed. Returning original.")
                return cv2_image_orig

            # Shape (num_detected_faces_in_image, 10)
            raw_landmarks_np_array = np.array(raw_landmarks_list_np)

            # Iterate over 10 landmark coords
            for i_lm in range(raw_landmarks_np_array.shape[1]):
                raw_landmarks_np_array[:, i_lm] = interpolate_sequence(raw_landmarks_np_array[:, i_lm])
            
            # Gaussian filter on a single item sequence doesn't change it much unless sigma is very small or padding is involved
            # For a single image, we can effectively skip the smoothing by using a small sigma or just reshaping
            # avg_landmarks_np_array = gaussian_filter1d(raw_landmarks_np_array, sigma=1, axis=0).reshape(raw_landmarks_np_array.shape[0], 5, 2)
            avg_landmarks_np_array = raw_landmarks_np_array.reshape(raw_landmarks_np_array.shape[0], 5, 2)
            avg_landmarks_list = [lm for lm in avg_landmarks_np_array]
        
        # Prepare Cropped Face Tensors
        cropped_face_t_list = []
        # This outer loop runs once for the single input image
        img_to_crop = input_img_list[0]
        
        if not has_aligned:
            if avg_landmarks_list is None or not avg_landmarks_list:
                print("Error: No landmarks available for alignment. Returning original.")
                return img_to_crop
            self.face_helper.clean_all()
            self.face_helper.read_image(img_to_crop)
            self.face_helper.all_landmarks_5 = [avg_landmarks_list[0]]
            self.face_helper.align_warp_face()
        else: # has_aligned = True
            img_resized = cv2.resize(img_to_crop, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.is_gray = is_gray(img_resized, threshold=10)
            if self.face_helper.is_gray: print('Grayscale input: True')
            self.face_helper.cropped_faces = [img_resized]

        if not self.face_helper.cropped_faces:
            print("Warning: No cropped face obtained after alignment. Returning original.")
            return cv2_image_orig

        cropped_face_cv2 = self.face_helper.cropped_faces[0]
        cropped_face_t = img2tensor(cropped_face_cv2 / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t_list.append(cropped_face_t)
        
        if not cropped_face_t_list:
            print("No faces to process after tensor conversion. Returning original.")
            return cv2_image_orig

        cropped_faces_tensor = torch.stack(cropped_face_t_list, dim=0).unsqueeze(0).to(self.device) # (1, 1, C, H, W)

        # Restore Faces with KEEP Network
        # print('Restoring face(s)...')

        # SINGLE IMAGE ADAPTATION: Duplicate the frame to make a 2-frame "sequence"
        # Input to net is (batch_size, num_frames, C, H, W)
        # For a single image, num_frames=1. We duplicate it to make it 2.
        duplicated_input_frames = torch.cat([cropped_faces_tensor, cropped_faces_tensor], dim=1) # (1, 2, C, H, W)
        
        # The KEEP model's forward pass in keep_arch.py takes `x` and `need_upscale`
        # It handles flows internally.
        net_output = self.keep_net(duplicated_input_frames, need_upscale=False) # (1, 2, C, H, W)
        
        # Take the first frame of the output (as both input frames were identical)
        final_output_tensor = net_output[:, 0:1, ...].squeeze(0) # (1, C, H, W) -> (C,H,W) if batch size was 1
                                                                # Actually (num_cropped_faces, C, H, W)
                                                                # Squeeze(0) makes it (num_frames_in_output, C, H, W)
                                                                # Which is (1, C, H, W)
        
        restored_faces_cv2 = [tensor2img(x, rgb2bgr=True, min_max=(-1, 1)) for x in final_output_tensor]
        torch.cuda.empty_cache()

        # Paste Faces Back
        # print('Pasting face(s) back...')

        final_restored_image = None
        # The original unprocessed input image
        img_for_pasting = input_img_list[0]
        
        self.face_helper.clean_all()

        if has_aligned:
            # Input was already aligned, so we don't paste, output is the restored face
            # The self.face_helper.cropped_faces might be stale here, use the input has_aligned image.
            img_resized = cv2.resize(img_for_pasting, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.is_gray = is_gray(img_resized, threshold=10)
            # self.face_helper.cropped_faces = [img_resized] # Storing the *original* cropped face for reference
        else:
            # Need to re-align the original image to get affine matrices for pasting back
            if avg_landmarks_list is None or not avg_landmarks_list: return img_for_pasting
            self.face_helper.read_image(img_for_pasting)
            self.face_helper.all_landmarks_5 = [avg_landmarks_list[0]]
            self.face_helper.align_warp_face() # This sets up affine matrices

        if not restored_faces_cv2 or not restored_faces_cv2[0].any():
            # print(f"Warning: Restored face is missing or invalid. Returning original.")
            return cv2_image_orig

        self.face_helper.add_restored_face(restored_faces_cv2[0].astype('uint8'))

        if not has_aligned:
            bg_img_upscaled = None
            if self.bg_upsampler is not None:
                # print("Upsampling background...")
                bg_img_upscaled = self.bg_upsampler.enhance(img_for_pasting, outscale=upscale_factor)[0]

            # Use stored affine_matrices
            self.face_helper.get_inverse_affine(None)
            
            current_face_upsampler = self.face_upsampler if self.face_upsampler is not None else None
            
            final_restored_image = self.face_helper.paste_faces_to_input_image(
                upsample_img=bg_img_upscaled, 
                draw_box=draw_box, 
                face_upsampler=current_face_upsampler
            )
        else: # has_aligned input, output is the restored face itself
            final_restored_image = self.face_helper.restored_faces[0]
            if self.face_upsampler and upscale_factor > 1:
                 # RealESRGANer is typically x2, so outscale must match if used
                 # The upscale_factor arg is for background. Face upsampler uses its internal scale.
                # print("Upsampling aligned face output...")
                final_restored_image = self.face_upsampler.enhance(final_restored_image, outscale=self.face_upsampler.scale)[0]
        
        if final_restored_image is None:
            return cv2_image_orig

        return final_restored_image

    @torch.no_grad()
    def process_video(self, video_path: str, output_video_path: str,
                      has_aligned: bool, only_center_face: bool, draw_box: bool,
                      upscale_factor: int = 1, max_clip_length: int = 20, target_fps: float = None):

        vidreader = VideoReader(video_path)
        original_fps = vidreader.get_fps()
        video_width = vidreader.width
        video_height = vidreader.height

        input_frames_bgr = []
        for _ in tqdm(range(len(vidreader))):
            img = vidreader.get_frame()
            if img is None:
                break
            input_frames_bgr.append(img)
        vidreader.close()

        if not input_frames_bgr:
            # print("No frames read from video.")
            return None
        
        num_input_frames = len(input_frames_bgr)
        # print(f"Total frames to process: {num_input_frames}")

        # This is done once for the whole video for temporal consistency
        avg_landmarks_sequence = None
        if not has_aligned:
            # print('Detecting keypoints and smoothing alignment for the whole video...')
            raw_landmarks_list_video = []
            for i in tqdm(range(num_input_frames), desc="Landmark Detection"):
                img = input_frames_bgr[i]
                self.face_helper.clean_all()
                self.face_helper.read_image(img)
                num_det_faces = self.face_helper.get_face_landmarks_5(
                    only_center_face=only_center_face, resize=640, eye_dist_threshold=5,
                    only_keep_largest=True)

                if num_det_faces >= 1:
                    raw_landmarks_list_video.append(self.face_helper.all_landmarks_5[0].reshape((10,)))
                else:
                    # Placeholder for missing faces
                    raw_landmarks_list_video.append(np.array([np.nan] * 10))

            # (num_frames, 10)
            raw_landmarks_np_video = np.array(raw_landmarks_list_video)
            
            if np.all(np.isnan(raw_landmarks_np_video)):
                # print("No faces detected in any frame. Cannot process video.")
                return None

            # Iterate over 10 landmark coordinates
            for i_lm_coord in range(raw_landmarks_np_video.shape[1]):
                raw_landmarks_np_video[:, i_lm_coord] = interpolate_sequence(raw_landmarks_np_video[:, i_lm_coord])
            
            # Smooth landmarks over time
            avg_landmarks_sequence = gaussian_filter1d(raw_landmarks_np_video, sigma=5, axis=0).reshape(num_input_frames, 5, 2)
            print("Landmark smoothing complete.")


        # Process frames in clips for KEEP network
        all_restored_faces_cv2 = [] # List to store restored face patches (cv2 format)
        # print("Cropping faces and preparing for KEEP network...")
        
        all_cropped_face_tensors = []
        for i in tqdm(range(num_input_frames), desc="Face Cropping"):
            img_bgr = input_frames_bgr[i]
            # Clean for each frame
            self.face_helper.clean_all()
            self.face_helper.read_image(img_bgr)

            if not has_aligned:
                if avg_landmarks_sequence is None or i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                    # print(f"Warning: Valid landmarks not available for frame {i}. Using previous frame's alignment or placeholder.")

                    # Fallback: use a placeholder or try to detect individually (less ideal for consistency)
                    # For simplicity here, if landmarks are bad, the cropped face might be bad.
                    if avg_landmarks_sequence is not None and i < len(avg_landmarks_sequence):
                         self.face_helper.all_landmarks_5 = [avg_landmarks_sequence[i]]
                    else: # No landmarks, create a dummy tensor
                         all_cropped_face_tensors.append(torch.zeros(3, 512, 512, dtype=torch.float32, device=self.device) * 0.5) # Normalized to 0 for (0.5,0.5,0.5)
                         continue 
                else:
                    self.face_helper.all_landmarks_5 = [avg_landmarks_sequence[i]]
                self.face_helper.align_warp_face()
            else: # has_aligned video input
                img_resized = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
                self.face_helper.is_gray = is_gray(img_resized, threshold=10)
                # if self.face_helper.is_gray: print(f'Frame {i} is grayscale.') # Can be noisy
                self.face_helper.cropped_faces = [img_resized]
            
            if not self.face_helper.cropped_faces: # Alignment failed or no face
                # print(f"Warning: No cropped face for frame {i}. Using placeholder.")
                all_cropped_face_tensors.append(torch.zeros(3, 512, 512, dtype=torch.float32, device=self.device) * 0.5)
                continue

            cropped_face_cv2 = self.face_helper.cropped_faces[0]
            cropped_face_t = img2tensor(cropped_face_cv2 / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            all_cropped_face_tensors.append(cropped_face_t)
        
        # Stack all cropped faces: (1, num_input_frames, C, H, W)
        if not all_cropped_face_tensors:
            # print("No faces could be cropped from the video.")
            return None
            
        batched_cropped_faces = torch.stack(all_cropped_face_tensors, dim=0).unsqueeze(0).to(self.device)

        print("Restoring faces with KEEP network (in clips)...")
        temp_restored_face_tensors = [] # Tensors from network
        for start_idx in tqdm(range(0, num_input_frames, max_clip_length), desc="KEEP Processing Clips"):
            end_idx = min(start_idx + max_clip_length, num_input_frames)
            current_clip_faces = batched_cropped_faces[:, start_idx:end_idx, ...]

            # Single frame in clip, duplicate for KEEP
            if current_clip_faces.shape[1] == 1:
                 current_clip_faces_for_net = torch.cat([current_clip_faces, current_clip_faces], dim=1)
                 clip_output_tensor = self.keep_net(current_clip_faces_for_net, need_upscale=False)
                 # Take first frame
                 temp_restored_face_tensors.append(clip_output_tensor[:, 0:1, ...])
            elif current_clip_faces.shape[1] > 1 :
                 clip_output_tensor = self.keep_net(current_clip_faces, need_upscale=False)
                 temp_restored_face_tensors.append(clip_output_tensor)
            # If current_clip_faces.shape[1] is 0, skip (should not happen with good indexing)
        
        if not temp_restored_face_tensors:
            # print("KEEP network processing yielded no results.")
            return None

        # (num_frames, C, H, W)
        all_restored_face_tensors = torch.cat(temp_restored_face_tensors, dim=1).squeeze(0)
        
        all_restored_faces_cv2 = [tensor2img(t, rgb2bgr=True, min_max=(-1, 1)) for t in all_restored_face_tensors]
        del batched_cropped_faces, all_restored_face_tensors, temp_restored_face_tensors
        torch.cuda.empty_cache()
        # print("KEEP network processing complete.")

        # Paste faces back and write video
        final_fps = target_fps if target_fps is not None else original_fps
        # Use dimensions of the first frame after potential BG upscaling for VideoWriter
        # Perform one BG upscale to determine output size if bg_upsampler is used
        
        temp_first_frame_for_size = input_frames_bgr[0]
        if self.bg_upsampler is not None:
            temp_first_frame_for_size = self.bg_upsampler.enhance(temp_first_frame_for_size, outscale=upscale_factor)[0]
        
        output_video_height, output_video_width = temp_first_frame_for_size.shape[0:2]
        del temp_first_frame_for_size

        vidwriter = VideoWriter(output_video_path, output_video_height, output_video_width, final_fps)
        
        # print("Pasting faces back and writing video...")
        for i in tqdm(range(num_input_frames), desc="Pasting & Writing Video"):
            original_frame_bgr = input_frames_bgr[i]

            # Should not happen if lists are aligned
            if i >= len(all_restored_faces_cv2):
                # print(f"Warning: Missing restored face for frame {i}. Using original frame.")
                # Or a black frame etc.
                final_frame_to_write = original_frame_bgr
                # Still upscale background if requested
                if self.bg_upsampler:
                    final_frame_to_write = self.bg_upsampler.enhance(final_frame_to_write, outscale=upscale_factor)[0]
                if final_frame_to_write.shape[0] != output_video_height or final_frame_to_write.shape[1] != output_video_width:
                    final_frame_to_write = cv2.resize(final_frame_to_write, (output_video_width, output_video_height))
                vidwriter.write_frame(final_frame_to_write)
                continue

            restored_face_patch_cv2 = all_restored_faces_cv2[i]

            self.face_helper.clean_all()
            self.face_helper.read_image(original_frame_bgr) # Set current frame in helper

            # Apply face_upsampler to the restored patch if needed
            current_restored_face_for_paste = restored_face_patch_cv2.astype('uint8')
            if self.face_upsampler:
                # If main background isn't upscaled (upscale_factor == 1), resize face back down after upsampling
                # to maintain geometric consistency, but with enhanced quality.
                if upscale_factor == 1 and getattr(self.face_upsampler, 'scale', 1) > 1:
                    effective_face_upsampler_scale = getattr(self.face_upsampler, 'scale', 2)
                    upscaled_face_content = self.face_upsampler.enhance(current_restored_face_for_paste, outscale=effective_face_upsampler_scale)[0]
                    
                    target_h_for_warp, target_w_for_warp = self.face_size[1], self.face_size[0] # H, W
                    current_restored_face_for_paste = cv2.resize(
                        upscaled_face_content, (target_w_for_warp, target_h_for_warp), 
                        interpolation=cv2.INTER_LANCZOS4
                    )
                else: # Background is also upscaled or no specific resize-down needed
                    effective_face_upsampler_scale = getattr(self.face_upsampler, 'scale', 2) # Default to 2
                    current_restored_face_for_paste = self.face_upsampler.enhance(current_restored_face_for_paste, outscale=effective_face_upsampler_scale)[0]
            
            self.face_helper.add_restored_face(current_restored_face_for_paste)

            if not has_aligned:
                if avg_landmarks_sequence is None or i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                    # print(f"Warning: Valid landmarks not available for pasting frame {i}. Using original frame content.")
                    final_frame_to_write = original_frame_bgr
                    if self.bg_upsampler: final_frame_to_write = self.bg_upsampler.enhance(final_frame_to_write, outscale=upscale_factor)[0]
                    if final_frame_to_write.shape[0] != output_video_height or final_frame_to_write.shape[1] != output_video_width:
                         final_frame_to_write = cv2.resize(final_frame_to_write, (output_video_width, output_video_height))
                    vidwriter.write_frame(final_frame_to_write)
                    continue
                self.face_helper.all_landmarks_5 = [avg_landmarks_sequence[i]] # Set current frame's landmarks
                self.face_helper.align_warp_face() # This is only to re-init affine matrix for this frame
                
                bg_img_upscaled_final = None
                if self.bg_upsampler is not None:
                    bg_img_upscaled_final = self.bg_upsampler.enhance(original_frame_bgr, outscale=upscale_factor)[0]
                
                self.face_helper.get_inverse_affine(None)
                pasted_frame = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img_upscaled_final, # This is the target canvas
                    draw_box=draw_box,
                    face_upsampler=None # Already handled current_restored_face_for_paste
                )
            else: # Video has_aligned (input frames are 512x512 aligned faces)
                # The output is just the restored (and potentially face-upsampled) face itself
                # Which is current_restored_face_for_paste
                pasted_frame = self.face_helper.restored_faces[0]
                # Ensure it's BGR
                if len(pasted_frame.shape) == 2: pasted_frame = cv2.cvtColor(pasted_frame, cv2.COLOR_GRAY2BGR)
            
            # Ensure frame is correct size for video writer
            if pasted_frame.shape[0] != output_video_height or pasted_frame.shape[1] != output_video_width:
                pasted_frame = cv2.resize(pasted_frame, (output_video_width, output_video_height), interpolation=cv2.INTER_LANCZOS4)

            vidwriter.write_frame(pasted_frame)
        
        vidwriter.close()
        print(f"Processed video saved to {output_video_path}")
        return output_video_path

    @torch.no_grad()
    def process_image_sequence(self,
                               image_sequence_tensor: torch.Tensor, # Input: B, H, W, C (ComfyUI IMAGE)
                               has_aligned_frames: bool,
                               only_center_face: bool,
                               draw_box: bool,
                               upscale_factor: int = 1, # For background
                               max_clip_length: int = 20):

        num_input_frames = image_sequence_tensor.shape[0]
        if num_input_frames == 0:
            return image_sequence_tensor # Return empty if input is empty

        # Convert ComfyUI IMAGE tensor to a list of cv2 BGR images
        input_frames_bgr = []
        for i in range(num_input_frames):
            # comfy_image_to_cv2 expects (1,H,W,C) or (H,W,C)
            frame_tensor = image_sequence_tensor[i].unsqueeze(0) 
            input_frames_bgr.append(comfy_image_to_cv2(frame_tensor))
        
        # print(f"Processing image sequence: {num_input_frames} frames.")

        # Landmark Detection and Smoothing (if not has_aligned)
        avg_landmarks_sequence = None
        if not has_aligned_frames:
            # print('Detecting keypoints and smoothing alignment for the sequence...')
            raw_landmarks_list_video = []
            for i in tqdm(range(num_input_frames), desc="Landmark Detection (Sequence)"):
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
                print("No faces detected in any frame. Returning original sequence.")
                return image_sequence_tensor 
            for i_lm_coord in range(raw_landmarks_np_video.shape[1]):
                raw_landmarks_np_video[:, i_lm_coord] = interpolate_sequence(raw_landmarks_np_video[:, i_lm_coord])
            avg_landmarks_sequence = gaussian_filter1d(raw_landmarks_np_video, sigma=5, axis=0).reshape(num_input_frames, 5, 2)
            # print("Landmark smoothing complete.")

        # Crop faces and prepare for KEEP network
        all_cropped_face_tensors = []
        # print("Cropping faces for KEEP network...")
        for i in tqdm(range(num_input_frames), desc="Face Cropping (Sequence)"):
            img_bgr = input_frames_bgr[i]
            self.face_helper.clean_all()
            self.face_helper.read_image(img_bgr)

            if not has_aligned_frames:
                if avg_landmarks_sequence is None or i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                    # Fallback for missing landmarks in a frame
                    print(f"Warning: Valid landmarks not available for frame {i} in sequence. Using placeholder crop.")
                    all_cropped_face_tensors.append(torch.zeros(3, 512, 512, dtype=torch.float32, device=self.device) * 0.5)
                    continue
                self.face_helper.all_landmarks_5 = [avg_landmarks_sequence[i]]
                self.face_helper.align_warp_face()
            else: # has_aligned_frames = True
                img_resized = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
                self.face_helper.is_gray = is_gray(img_resized, threshold=10)
                self.face_helper.cropped_faces = [img_resized]
            
            if not self.face_helper.cropped_faces:
                print(f"Warning: No cropped face for frame {i}. Using placeholder crop.")
                all_cropped_face_tensors.append(torch.zeros(3, 512, 512, dtype=torch.float32, device=self.device) * 0.5)
                continue

            cropped_face_cv2 = self.face_helper.cropped_faces[0]
            cropped_face_t = img2tensor(cropped_face_cv2 / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            all_cropped_face_tensors.append(cropped_face_t)
        
        if not all_cropped_face_tensors:
            print("No faces could be cropped from the image sequence.")
            return image_sequence_tensor # Return original

        # Batch for KEEP network: (1, num_input_frames, C, H, W)
        batched_cropped_faces = torch.stack(all_cropped_face_tensors, dim=0).unsqueeze(0).to(self.device)

        # KEEP Network Processing (in clips)
        # print("Restoring faces with KEEP network (in clips)...")
        temp_restored_face_tensors = []
        for start_idx in tqdm(range(0, num_input_frames, max_clip_length), desc="KEEP Processing Clips (Sequence)"):
            end_idx = min(start_idx + max_clip_length, num_input_frames)
            current_clip_faces = batched_cropped_faces[:, start_idx:end_idx, ...]
            
            if current_clip_faces.shape[1] == 0: continue # Skip empty clips
            if current_clip_faces.shape[1] == 1:
                 current_clip_faces_for_net = torch.cat([current_clip_faces, current_clip_faces], dim=1)
                 clip_output_tensor = self.keep_net(current_clip_faces_for_net, need_upscale=False)
                 temp_restored_face_tensors.append(clip_output_tensor[:, 0:1, ...])
            else:
                 clip_output_tensor = self.keep_net(current_clip_faces, need_upscale=False)
                 temp_restored_face_tensors.append(clip_output_tensor)
        
        if not temp_restored_face_tensors:
            print("KEEP network processing yielded no results for the sequence.")
            return image_sequence_tensor

        all_restored_face_tensors = torch.cat(temp_restored_face_tensors, dim=1).squeeze(0)
        all_restored_faces_cv2 = [tensor2img(t, rgb2bgr=True, min_max=(-1, 1)) for t in all_restored_face_tensors]
        del batched_cropped_faces, all_restored_face_tensors, temp_restored_face_tensors
        torch.cuda.empty_cache()
        print("KEEP network processing complete for sequence.")

        # Paste faces back onto each frame
        output_frames_cv2 = []
        # print("Pasting faces back onto frames...")
        for i in tqdm(range(num_input_frames), desc="Pasting Faces (Sequence)"):
            original_frame_bgr = input_frames_bgr[i]
            
            if i >= len(all_restored_faces_cv2):
                print(f"Warning: Missing restored face for frame {i}. Using original frame content.")
                final_pasted_frame = original_frame_bgr
                if self.bg_upsampler:
                    final_pasted_frame = self.bg_upsampler.enhance(final_pasted_frame, outscale=upscale_factor)[0]
                output_frames_cv2.append(final_pasted_frame)
                continue

            restored_face_patch_cv2 = all_restored_faces_cv2[i]

            self.face_helper.clean_all()
            self.face_helper.read_image(original_frame_bgr)

            current_restored_face_for_paste = restored_face_patch_cv2.astype('uint8')
            if self.face_upsampler:
                if upscale_factor == 1 and getattr(self.face_upsampler, 'scale', 1) > 1:
                    effective_face_upsampler_scale = getattr(self.face_upsampler, 'scale', 2)
                    upscaled_face_content = self.face_upsampler.enhance(current_restored_face_for_paste, outscale=effective_face_upsampler_scale)[0]
                    
                    # ------ some fix here
                    # self.face_size is actually self.face_helper.face_size
                    # self.face_helper.face_size is (width, height)
                    target_w_for_warp, target_h_for_warp = self.face_helper.face_size[0], self.face_helper.face_size[1]
                    # ------ end of fix
                    
                    current_restored_face_for_paste = cv2.resize(
                        upscaled_face_content, (target_w_for_warp, target_h_for_warp), 
                        interpolation=cv2.INTER_LANCZOS4)
                else:
                    effective_face_upsampler_scale = getattr(self.face_upsampler, 'scale', 2)
                    current_restored_face_for_paste = self.face_upsampler.enhance(current_restored_face_for_paste, outscale=effective_face_upsampler_scale)[0]
            
            self.face_helper.add_restored_face(current_restored_face_for_paste)

            if not has_aligned_frames:
                if avg_landmarks_sequence is None or i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                    print(f"Warning: Valid landmarks not available for pasting frame {i}. Using original content.")
                    final_pasted_frame = original_frame_bgr
                    if self.bg_upsampler: final_pasted_frame = self.bg_upsampler.enhance(final_pasted_frame, outscale=upscale_factor)[0]
                    output_frames_cv2.append(final_pasted_frame)
                    continue
                self.face_helper.all_landmarks_5 = [avg_landmarks_sequence[i]]
                self.face_helper.align_warp_face() 
                
                bg_img_upscaled_final = None
                if self.bg_upsampler is not None:
                    bg_img_upscaled_final = self.bg_upsampler.enhance(original_frame_bgr, outscale=upscale_factor)[0]
                
                self.face_helper.get_inverse_affine(None)
                final_pasted_frame = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img_upscaled_final,
                    draw_box=draw_box,
                    face_upsampler=None 
                )
            else: # has_aligned_frames = True
                final_pasted_frame = self.face_helper.restored_faces[0]
                if len(final_pasted_frame.shape) == 2: 
                    final_pasted_frame = cv2.cvtColor(final_pasted_frame, cv2.COLOR_GRAY2BGR)

            output_frames_cv2.append(final_pasted_frame)
            
        # Convert list of cv2 BGR images back to ComfyUI IMAGE tensor (B, H, W, C), RGB, float32 [0,1]
        output_tensors = [cv2_to_comfy_image(frame) for frame in output_frames_cv2]
        if not output_tensors:
             return image_sequence_tensor
        final_batch_tensor = torch.cat(output_tensors, dim=0)
        print("Video frame processing complete.")
        return final_batch_tensor