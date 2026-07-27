import logging
from src.datamodules.base import BaseDataModule
from src.datasets.diode import DIODEDataset


log = logging.getLogger(__name__)


class DIODEDataModule(BaseDataModule):
    """LightningDataModule for DIODE dataset."""

    _DATASET_CLASS = DIODEDataset
    _MINIDATASET_CLASS = DIODEDataset
