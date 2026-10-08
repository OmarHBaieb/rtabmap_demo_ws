import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'transformer_object_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Copie le(s) modèle(s) TorchScript exporté(s) dans share/<pkg>/models au build.
        # Le fichier models/multimodal_transformer.pt doit exister avant colcon build
        # (généré via scripts/export_model.py).
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='omar',
    maintainer_email='omar.hbaieb2016@gmail.com',
    description='Détecteur de traversabilité RGB-D par Transformer multimodal '
                 '(DINOv2 + ViT depth + cross-attention), remplace find_object_2d '
                 'dans rtabmap_demo_ws.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detector_node = transformer_object_detector.detector_node:main',
            'fusion_node = transformer_object_detector.fusion_node:main',
        ],
    },
)