# rtabmap_demo_ws

Segmentation de traversabilité RGB-D par Transformer multimodal pour un robot mobile (ROS 2 Humble, Gazebo Classic, RTAB-Map, Nav2).

## Principe

Le modèle classe chaque pixel en trois classes : sol, obstacle, inconnu.

- RGB : DINOv2-S/14 (gelé, sauf les 2 derniers blocs lors du fine-tuning Gazebo)
- Profondeur : petit ViT entraîné from scratch
- Fusion par cross-attention, puis tête de segmentation 3 classes

Deux chemins parallèles, qui ne se rejoignent qu'au niveau de Nav2 :

1. Caméra RGB-D -> `detector_node` (Transformer) -> `fusion_node` -> nuages `points_ground` / `points_obstacles` -> costmap local Nav2 (`voxel_layer`)
2. Caméra RGB-D -> RTAB-Map (SLAM sur le RGB-D brut) -> Nav2 (carte et localisation)

## Résultats

| Modèle | mIoU NYUv2 (test) | mIoU Gazebo |
|---|---|---|
| Entraînement initial | 0,7638 | effondré (100 % obstacle) |
| Fine-tuné (epoch 13, déployé) | 0,7586 | 0,9802 |

Les étiquettes Gazebo sont générées géométriquement (reprojection de la profondeur et hauteur au-dessus du sol). Le mIoU Gazebo mesure donc l'accord avec la géométrie en simulation, pas une performance en conditions réelles.

## Installation

Prérequis : Ubuntu 22.04, ROS 2 Humble, Gazebo Classic, paquets TurtleBot3, PyTorch avec CUDA.

```bash
mkdir -p ~/rtabmap_demo_ws/src && cd ~/rtabmap_demo_ws/src
git clone https://github.com/OmarHBaieb/rtabmap_demo_ws.git .
git clone https://github.com/introlab/rtabmap.git
git clone -b ros2 https://github.com/introlab/rtabmap_ros.git
cd .. && colcon build --symlink-install
source install/setup.bash
```

Lance Gazebo, RViz2 et `colcon build` depuis un terminal natif, pas depuis le terminal intégré de VS Code (conflits de bibliothèques snap).

## Modèle

Les poids ne sont pas dans le dépôt. Télécharge `multimodal_transformer.pt` depuis la page Releases et place-le dans `src/transformer_object_detector/models/`.

## Lancement

Pour que Nav2 utilise la segmentation, remplace le fichier de paramètres de la démo par la version sémantique (garde une sauvegarde) :

```bash
P=src/rtabmap_ros/rtabmap_demos/params/humble/turtlebot3_rgbd_nav2_params.yaml
cp $P $P.backup
cp src/transformer_object_detector/config/turtlebot3_rgbd_nav2_params_semantic.yaml $P
colcon build --packages-select rtabmap_demos --symlink-install
```

Terminal 1 : simulation, SLAM et Nav2.

```bash
export TURTLEBOT3_MODEL=waffle
ros2 launch rtabmap_demos turtlebot3_sim_rgbd_demo.launch.py
```

Terminal 2 : pipeline Transformer.

```bash
ros2 launch transformer_object_detector transformer_pipeline.launch.py
```

Topics publiés : `/segmentation/mask`, `/segmentation/overlay`, `/segmentation/pointcloud`, `/segmentation/points_ground`, `/segmentation/points_obstacles`.

## Entraînement

Données NYUv2 dans `~/nyu_data/`, scripts dans `training/`.

```bash
cd training
python3 train.py --nyu_root ~/nyu_data/NYUv2 --masks_root ~/nyu_data/traversability \
  --epochs 50 --patience 8 --floor_boost 3.0 --output checkpoints/traversability_best.pth

python3 finetune_gazebo.py --checkpoint checkpoints/traversability_best.pth \
  --nyu_root ~/nyu_data/NYUv2 --nyu_masks_root ~/nyu_data/traversability \
  --unfreeze_last_n_blocks 2 --nyu_replay_ratio 2.0 --epochs 15 --lr 1e-5 \
  --output checkpoints/traversability_gazebo_finetuned.pth
```

Export TorchScript :

```bash
python3 src/transformer_object_detector/scripts/export_model.py \
  --checkpoint training/checkpoints/traversability_gazebo_finetuned.pth \
  --output src/transformer_object_detector/models/multimodal_transformer.pt --method trace
```

## Limites

- Fine-tuning sur 80 images d'un seul monde Gazebo : la généralisation à d'autres scènes n'est pas démontrée.
- Aucun test sur robot réel.
- Le nœud d'inférence tourne à environ 15 Hz sur une RTX 3050 6 Go.
