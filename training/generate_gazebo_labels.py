"""
Bootstrap géométrique des labels sol/obstacle/inconnu pour frames Gazebo.
Utilise la géométrie caméra connue (intrinsèques + extrinsèques TF) plutôt
qu'une annotation manuelle, car Gazebo fournit une depth exacte (non bruitée).
"""
import numpy as np
import cv2

# --- intrinsèques caméra (ros2 topic echo /camera/camera_info) ---
FX, FY = 565.6008952774197, 565.6008952774197
CX, CY = 320.5, 240.5

# --- extrinsèques caméra -> base_footprint (ros2 run tf2_ros tf2_echo) ---
R = np.array([
    [0.001,  0.001,  1.000],
    [-1.000, 0.000,  0.001],
    [0.000, -1.000,  0.001],
])
T = np.array([0.069, -0.047, 0.117])

# --- seuils de classification (en mètres, hauteur au-dessus du sol) ---
FLOOR_MAX_HEIGHT = 0.05    # < 5cm au-dessus du sol -> sol
OBSTACLE_MIN_HEIGHT = 0.10 # > 10cm -> obstacle
# entre les deux (5-10cm) ou depth invalide -> inconnu

DEPTH_MAX_M = 3.0  # doit matcher DEPTH_MAX_M du modèle


def generate_label(depth_img: np.ndarray) -> np.ndarray:
    h, w = depth_img.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))

    valid = (depth_img > 0.05) & (depth_img < DEPTH_MAX_M) & np.isfinite(depth_img)

    # nettoyage AVANT reprojection pour éviter tout nan/inf dans la matmul
    Z_cam = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
    X_cam = (us - CX) * Z_cam / FX
    Y_cam = (vs - CY) * Z_cam / FY

    # points caméra -> base_footprint (sol = z_base ≈ 0)
    pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)  # [H,W,3]
    pts_base = pts_cam @ R.T + T  # [H,W,3]
    height_above_ground = pts_base[..., 2]

    mask = np.full((h, w), 2, dtype=np.uint8)  # défaut: inconnu
    mask[valid & (height_above_ground < FLOOR_MAX_HEIGHT)] = 0       # sol
    mask[valid & (height_above_ground > OBSTACLE_MIN_HEIGHT)] = 1    # obstacle
    return mask


if __name__ == "__main__":
    import sys
    depth = np.load(sys.argv[1])  # ex: depth_frame.npy
    mask = generate_label(depth)
    out_path = sys.argv[2] if len(sys.argv) > 2 else "mask_preview.png"
    # visualisation: sol=vert, obstacle=rouge, inconnu=gris
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask == 0] = [0, 255, 0]
    vis[mask == 1] = [0, 0, 255]
    vis[mask == 2] = [128, 128, 128]
    cv2.imwrite(out_path, vis)
    print(f"sol: {(mask==0).sum()}, obstacle: {(mask==1).sum()}, inconnu: {(mask==2).sum()}")
    print(f"preview -> {out_path}")