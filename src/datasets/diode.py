import os
import glob
import torch
import numpy as np
import os.path as osp
from typing import List

from src.datasets.base import BaseDataset
from src.data import Data, InstanceData


class DIODEDataset(BaseDataset):
    """DIODE dataset wrapper for Superpoint Transformer."""

    def __init__(self, root: str, *args, **kwargs):
        # Map 'split' to 'stage' to ensure compatibility with split='val' calls
        if "split" in kwargs:
            kwargs["stage"] = kwargs.pop("split")
            
        # Dynamically detect if we want to run specifically on indoors or outdoor scenes
        stage = kwargs.get("stage", "train")
        self.scene_type = "all"
        for st in ["indoors", "outdoor"]:
            if stage.endswith(f"_{st}"):
                self.scene_type = st
                kwargs["stage"] = stage.replace(f"_{st}", "")
                break

        # Store the original root before BaseDataset modifies it
        self.dataset_root = os.path.abspath(root)
        super().__init__(root, *args, **kwargs)


    @property
    def data_subdir_name(self) -> str:
        # Override so self.root is exactly {dataset_root}/diode
        return "diode"

    @property
    def class_names(self) -> List[str]:
        return [
            "wall", "floor", "ceiling", "ground", "road",
            "sky", "building", "tree", "vegetation", "water",
            "mountain", "vehicle", "person", "chair", "table",
            "sofa", "cabinet", "window", "door", "other",
            "ignored",
        ]


    @property
    def num_classes(self) -> int:
        return 20

    @property
    def stuff_classes(self) -> List[int]:
        return []

    @property
    def class_colors(self) -> List[List[int]]:
        return [
            [200, 50, 50],    # 0  wall (red-brown)
            [130, 130, 200],  # 1  floor (blue-grey)
            [230, 230, 130],  # 2  ceiling (yellow)
            [140, 100, 60],   # 3  ground (brown)
            [100, 100, 100],  # 4  road (grey)
            [80, 180, 230],   # 5  sky (light blue)
            [150, 100, 50],   # 6  building (dark orange)
            [30, 140, 30],    # 7  tree (dark green)
            [100, 200, 100],  # 8  vegetation (light green)
            [30, 100, 200],   # 9  water (deep blue)
            [160, 140, 120],  # 10 mountain (tan)
            [230, 130, 30],   # 11 vehicle (orange)
            [230, 180, 180],  # 12 person (pink)
            [130, 50, 160],   # 13 chair (purple)
            [180, 120, 60],   # 14 table (wood brown)
            [200, 80, 120],   # 15 sofa (magenta)
            [80, 140, 140],   # 16 cabinet (teal)
            [200, 230, 230],  # 17 window (ice blue)
            [100, 50, 30],    # 18 door (dark brown)
            [180, 180, 180],  # 19 other (light grey)
            [0, 0, 0],        # 20 ignored (black)
        ]

    @property
    def raw_dir(self) -> str:
        # Return the folder containing the raw_ply directory
        # Use raw_ply_data for train/test splits if it exists, and raw_ply for val split
        if self.stage in ["train", "test"] and osp.exists(osp.join(self.dataset_root, "raw_ply_data")):
            return osp.join(self.dataset_root, "raw_ply_data")
        return osp.join(self.dataset_root, "raw_ply")

    @property
    def all_base_cloud_ids(self) -> dict:
        res = {"train": [], "val": [], "test": []}
        
        # Scan raw_ply directory recursively for PLY files
        if not osp.exists(self.raw_dir):
            # Fallback dummy IDs if raw_ply does not exist yet
            return {"train": [], "val": ["dummy_scene"], "test": []}

        for split in ["train", "val", "test"]:
            split_dir = osp.join(self.raw_dir, split)
            if osp.exists(split_dir):
                ply_files = sorted(glob.glob(osp.join(split_dir, "**", "*.ply"), recursive=True))
                ids = []
                for f in ply_files:
                    rel_path = osp.relpath(f, split_dir)
                    base_id = osp.splitext(rel_path)[0]
                    # Filter based on scene_type ("indoors", "outdoor", or "all")
                    if self.scene_type == "all" or self.scene_type in base_id:
                        ids.append(base_id)
                res[split] = ids

        # Ensure we have at least a dummy value for val to prevent PyG initialization errors
        if not res["val"]:
            res["val"] = ["dummy_scene"]

        return res

    def download_dataset(self) -> None:
        # Data is pre-downloaded and converted to PLY via load_diode.py
        pass

    def id_to_relative_raw_path(self, id: str) -> str:
        tile = self.get_tile_from_path(id)
        base_id = tile[1] if tile is not None else id
        return osp.join(self.stage, base_id + ".ply")

    def processed_to_raw_path(self, processed_path: str) -> str:
        # Dynamically map the processed path structure back to raw_ply
        prefix = osp.join(self.processed_dir, self.stage, self.pre_transform_hash)
        rel_path = osp.relpath(processed_path, prefix)
        base_id = osp.splitext(rel_path)[0]
        tile = self.get_tile_from_path(base_id)
        base_id = tile[1] if tile is not None else base_id
        raw_rel = base_id + ".ply"
        return osp.join(self.raw_dir, self.stage, raw_rel)

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        import open3d as o3d
        import logging

        log = logging.getLogger(__name__)

        # Load the point cloud
        pcd = o3d.io.read_point_cloud(raw_cloud_path)
        pos = torch.from_numpy(np.asarray(pcd.points)).float()
        colors = np.asarray(pcd.colors)
        rgb = torch.from_numpy(colors).float() if colors.size > 0 else torch.zeros(pos.shape[0], 3)

        # Guard against empty point clouds (0 points in PLY file).
        # InstanceData crashes on empty tensors because obj.max() is
        # undefined for numel()==0.  Return a minimal single-point
        # placeholder with a void label so the pipeline can proceed.
        if pos.shape[0] == 0:
            log.warning(
                f"Empty point cloud (0 points) detected: {raw_cloud_path}. "
                "Inserting a single dummy point with void label."
            )
            pos = torch.zeros(1, 3)
            rgb = torch.zeros(1, 3)
            y = torch.tensor([self.num_classes], dtype=torch.long)  # void label
            idx = torch.zeros(1, dtype=torch.long)
            obj = torch.zeros(1, dtype=torch.long)
            count = torch.ones(1, dtype=torch.long)
            instance_data = InstanceData(idx, obj, count, y, dense=True)
            data = Data(pos=pos, rgb=rgb, y=y)
            data.obj = instance_data
            return data

        # Load SegFormer-generated semantic labels
        label_path = raw_cloud_path.replace(".ply", "_labels.npy")
        if os.path.isfile(label_path):
            y = torch.from_numpy(np.load(label_path).astype(np.int64))
            # Ensure labels are valid (0 to num_classes-1), map anything else to ignored
            y[y < 0] = self.num_classes
            y[y >= self.num_classes] = self.num_classes
        else:
            log.warning(f"No label file found: {label_path}, using void labels")
            y = torch.full((pos.shape[0],), self.num_classes, dtype=torch.long)

        # Set up a dummy InstanceData object (some models/dataloaders require data.obj)
        idx = torch.arange(pos.shape[0])
        obj = torch.zeros(pos.shape[0], dtype=torch.long)
        count = torch.ones_like(obj)
        instance_data = InstanceData(idx, obj, count, y, dense=True)

        data = Data(pos=pos, rgb=rgb, y=y)
        data.obj = instance_data

        return data
