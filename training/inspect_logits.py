import sys, torch, numpy as np
sys.path.insert(0, '/home/omar/Downloads/rtabmap_demo_ws/src/transformer_object_detector/transformer_object_detector')

from detector_node import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, DEPTH_MAX_M

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from PIL import Image as PILImage
import cv2
def preprocess_rgb(rgb_img):
    rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    rgb = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(rgb).unsqueeze(0).to(device)

def preprocess_depth(depth_img):
    is_uint_mm = depth_img.dtype != np.float32
    depth = depth_img.astype(np.float32)
    if is_uint_mm:
        depth = depth / 1000.0
    depth = np.nan_to_num(depth, nan=0.0, posinf=DEPTH_MAX_M, neginf=0.0)
    depth[depth > 50.0] = DEPTH_MAX_M
    depth = np.clip(depth, 0.0, DEPTH_MAX_M)
    depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    depth = depth / DEPTH_MAX_M
    return torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)

# --- chargement modèle + frames capturées ---
model = torch.jit.load('/home/omar/Downloads/rtabmap_demo_ws/src/transformer_object_detector/models/multimodal_transformer.pt', map_location=device).eval()
name = "00001.png"
rgb_pil = PILImage.open(f'/home/omar/nyu_data/NYUv2/image/test/{name}').convert('RGB')
rgb_img = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)

depth_img = np.array(PILImage.open(f'/home/omar/nyu_data/NYUv2/depth/test/{name}'))

rgb_t = preprocess_rgb(rgb_img)
depth_t = preprocess_depth(depth_img)

with torch.no_grad():
    logits = model(rgb_t, depth_t)  # [1, 3, H, W]
with torch.no_grad():
    logits_rgb_only = model(rgb_t, torch.zeros_like(depth_t))
    logits_depth_only = model(torch.zeros_like(rgb_t), depth_t)

for name, l in [("RGB seul", logits_rgb_only), ("Depth seul", logits_depth_only)]:
    gap = (l[0,1] - l[0,0])
    print(f"{name} -> mean gap: {gap.mean().item():.3f}, min: {gap.min().item():.3f}, max: {gap.max().item():.3f}")

# ablation croisée : RGB Gazebo + depth NYUv2, et inverse
rgb_gazebo = preprocess_rgb(np.load('../rgb_frame.npy'))
depth_gazebo = preprocess_depth(np.load('../depth_frame.npy'))

rgb_nyu = preprocess_rgb(rgb_img)      # rgb_img = NYUv2 (déjà chargé au-dessus)
depth_nyu = preprocess_depth(depth_img)

with torch.no_grad():
    logits_mix1 = model(rgb_gazebo, depth_nyu)   # RGB Gazebo + depth NYUv2
    logits_mix2 = model(rgb_nyu, depth_gazebo)   # RGB NYUv2 + depth Gazebo

for name, l in [("RGB-Gazebo + Depth-NYU", logits_mix1), ("RGB-NYU + Depth-Gazebo", logits_mix2)]:
    gap = (l[0,1] - l[0,0])
    print(f"{name} -> mean: {gap.mean().item():.3f}, min: {gap.min().item():.3f}, max: {gap.max().item():.3f}")
with torch.no_grad():
    logits_gazebo_full = model(rgb_gazebo, depth_gazebo)
gap_gazebo = (logits_gazebo_full[0,1] - logits_gazebo_full[0,0])
print(f"Gazebo COMPLET (RGB+Depth reels) -> mean: {gap_gazebo.mean().item():.3f}, min: {gap_gazebo.min().item():.3f}, max: {gap_gazebo.max().item():.3f}")
print("shape:", logits.shape)

print("mean sol      :", logits[0,0].mean().item())
print("mean obstacle :", logits[0,1].mean().item())
print("mean unknown  :", logits[0,2].mean().item())
print("min gap (obstacle-sol):", (logits[0,1]-logits[0,0]).min().item())
print("max gap (obstacle-sol):", (logits[0,1]-logits[0,0]).max().item())