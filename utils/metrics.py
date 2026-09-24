import numpy as np
from PIL import Image
import utils.spritegeom as G
import lpips, torch


def to_tensor(img):
    a = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    return t * 2.0 - 1.0


def run_metrics(truth , generated):

    # Convert RGB 
    truth_img = G.flatten(Image.open(truth))
    generated_img = {k: G.flatten(Image.open(path)) for k, path in generated.items()}
    
    # ==========================================================================
    # LPIPS
    # ==========================================================================    
    loss_fn = lpips.LPIPS(net='alex')
    t_tensor = to_tensor(truth_img)
    lpips_value = {}

   
    for k, img in generated_img.items():
        if img.size != truth_img.size:
            raise ValueError(f"{k}: {img.size} != truth {truth_img.size}")
        lpips_value[k] = loss_fn(to_tensor(img), t_tensor).item()

    # ==========================================================================
    # MAE
    # ==========================================================================
    # Convert PIL to NumPY 
    arr_truth = np.array(truth_img, dtype=np.float32)
    mae_value = {}

    for k, img in generated_img.items():
        if img.size != truth_img.size:
            raise ValueError(f"{k}: {img.size} != truth {truth_img.size}")
        mae_value[k] = float(np.abs(arr_truth - np.array(img, np.float32)).mean())


    return {
        "LPIPS" : lpips_value,
        "MAE"   : mae_value
    }



