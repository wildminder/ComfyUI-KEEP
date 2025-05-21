import os
import torch

try:
    from basicsr.utils.download_util import load_file_from_url
except ImportError:
    # Minimal fallback if basicsr isn't in path when facelib is directly used (should not happen with our setup)
    def load_file_from_url(url, model_dir, **kwargs):
        print(f"Fallback load_file_from_url in facelib.detection for: {url} to {model_dir}")
        filename = os.path.basename(urlparse(url).path)
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path): raise FileNotFoundError(f"{path} not found.")
        return path

# Model URLs (these might be defined elsewhere in original facelib, ensure they are here or imported)
URLS = {
    'retinaface_resnet50': 'https://github.com/jnjaby/KEEP/releases/download/v1.0.0/detection_Resnet50_Final.pth',
    'retinaface_mobile0.25': 'https://github.com/jnjaby/KEEP/releases/download/v1.0.0/detection_mobilenet0.25_Final.pth',
    # Add Yolo URLs if init_detection_model handles them
    'yolov5n': 'https://github.com/jnjaby/KEEP/releases/download/v1.0.0/yolov5n-face.pth', # Example
    'yolov5l': 'https://github.com/jnjaby/KEEP/releases/download/v1.0.0/yolov5l-face.pth', # Example
}
MODEL_NAMES = { # Map cli name to filename
    'retinaface_resnet50': 'detection_Resnet50_Final.pth',
    'retinaface_mobile0.25': 'detection_mobilenet0.25_Final.pth',
    'YOLOv5n': 'yolov5n-face.pth',
    'YOLOv5l': 'yolov5l-face.pth',
}


def init_detection_model(model_name, half=False, device='cuda', model_rootpath=None): # <<< ADDED model_rootpath
    if model_name == 'retinaface_resnet50':
        from .retinaface.retinaface import RetinaFace
        model = RetinaFace(network_name='resnet50', half=half)
        model_file_name = MODEL_NAMES['retinaface_resnet50']
    elif model_name == 'retinaface_mobile0.25':
        from .retinaface.retinaface import RetinaFace
        model = RetinaFace(network_name='mobile0.25', half=half)
        model_file_name = MODEL_NAMES['retinaface_mobile0.25']
    elif model_name == 'YOLOv5l':
        from .yolov5face.face_detector import YoloDetector
        model = YoloDetector(config_name='facelib/detection/yolov5face/models/yolov5l.yaml', device=device)
        model_file_name = MODEL_NAMES['YOLOv5l']
    elif model_name == 'YOLOv5n':
        from .yolov5face.face_detector import YoloDetector
        model = YoloDetector(config_name='facelib/detection/yolov5face/models/yolov5n.yaml', device=device)
        model_file_name = MODEL_NAMES['YOLOv5n']
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    if model_rootpath is None:
        # Fallback to a default relative path if no root is provided
        # For ComfyUI, model_rootpath should ALWAYS be provided by the caller (KEEPModelLoader)
        model_path_final = load_file_from_url(url=URLS[model_name], model_dir='weights/facelib', file_name=model_file_name)
        print(f"Warning (init_detection_model): model_rootpath is None. Using default relative path: {model_path_final}")
    else:
        model_path_final = os.path.join(model_rootpath, model_file_name)

    if not os.path.exists(model_path_final):
        # This case should ideally be handled by the ComfyUI loader downloading it first.
        # If it still doesn't exist, it means the ComfyUI loader failed or paths are wrong.
        print(f"Warning (init_detection_model): Model file not found at {model_path_final}. Attempting download via facelib's URL.")
        # This load_file_from_url is from basicsr, which should be in sys.path
        # It will try to download to a generic 'weights/facelib' if model_dir is relative like this.
        # This is not ideal as it bypasses ComfyUI's model management.
        # Better to ensure load_file_from_url_comfy in our utils handles all downloads.
        model_path_final = load_file_from_url(url=URLS[model_name], model_dir=model_rootpath, file_name=model_file_name)


    load_net = torch.load(model_path_final, map_location=lambda storage, loc: storage)
    if 'state_dict' in load_net:
        load_net = load_net['state_dict']
    
    # Special handling for YOLO models from the original inference_keep.py
    if model_name in ['YOLOv5l', 'YOLOv5n']:
        if "model_state_dict" in load_net: # Check if it's the expected ultralytics format
            model.model.load_state_dict(load_net["model_state_dict"])
        else: # Assume it's a raw state_dict then
            model.model.load_state_dict(load_net) # or model.load_state_dict(load_net) if YoloDetector handles it
    else: # RetinaFace
        new_load_net = {}
        for k, v in load_net.items():
            new_load_net[k.replace('module.', '')] = v
        model.load_state_dict(new_load_net)

    model.eval()
    model = model.to(device)
    return model