# File: ComfyUI-KEEP/modules/deps/facelib/parsing/__init__.py
import os
import torch
# Assuming load_file_from_url is now globally accessible via sys.path or vendored
try:
    from wm_basicsr.utils.download_util import load_file_from_url
except ImportError:
    def load_file_from_url(url, model_dir, **kwargs):
        print(f"Fallback load_file_from_url in facelib.parsing for: {url} to {model_dir}")
        filename = os.path.basename(urlparse(url).path)
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path): raise FileNotFoundError(f"{path} not found.")
        return path
        
URLS = {
    'parsenet': 'https://github.com/jnjaby/KEEP/releases/download/v1.0.0/parsing_parsenet.pth'
}
MODEL_NAMES = {
    'parsenet': 'parsing_parsenet.pth'
}

def init_parsing_model(model_name='parsenet', half=False, device='cuda', model_rootpath=None): # <<< ADDED model_rootpath
    if model_name == 'parsenet':
        from .parsenet import ParseNet
        model = ParseNet(in_size=512, out_size=512)
        model_file_name = MODEL_NAMES['parsenet']
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    # ---- MODIFIED MODEL LOADING ----
    if model_rootpath is None:
        model_path_final = load_file_from_url(url=URLS[model_name], model_dir='weights/facelib', file_name=model_file_name)
        print(f"Warning (init_parsing_model): model_rootpath is None. Using default relative path: {model_path_final}")
    else:
        model_path_final = os.path.join(model_rootpath, model_file_name)
    
    if not os.path.exists(model_path_final):
        print(f"Warning (init_parsing_model): Model file not found at {model_path_final}. Attempting download via facelib's URL.")
        model_path_final = load_file_from_url(url=URLS[model_name], model_dir=model_rootpath, file_name=model_file_name)

    load_net = torch.load(model_path_final, map_location=lambda storage, loc: storage)
    # remove prefix 'module.'
    new_load_net = {}
    for k, v in load_net.items():
        new_load_net[k.replace('module.', '')] = v
    model.load_state_dict(new_load_net)
    # ---- END OF MODIFIED MODEL LOADING ----
    
    model.eval()
    model = model.to(device)
    return model