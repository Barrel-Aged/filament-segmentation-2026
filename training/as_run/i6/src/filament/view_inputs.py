"""Full-image view reader: verify hashes once, then reject changed file metadata."""
from pathlib import Path
import numpy as np

from .common import read_json
from .views import cache_views


def file_state(path):
    stat = Path(path).stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class CachedViewReader:
    def __init__(self, channels, config="configs/views-v1.json", root="data/derived"):
        self.channels = tuple(channels)
        if not channels or channels[0] != "raw" or len(set(channels)) != len(channels):
            raise ValueError("Unique channels must start with raw")
        self.config, self.root = Path(config), Path(root)
        self.paths, self.verified = {}, {}
        self.read_only = False

    def __getstate__(self):
        # Spawned workers verify content independently; do not pickle memory maps.
        return {**self.__dict__, "verified": {}}

    def load(self, image_path):
        image_path = Path(image_path)
        key = str(image_path)
        if key not in self.verified:
            if self.read_only:
                if key not in self.paths:
                    raise ValueError("View was not prewarmed")
                expected = Path(self.paths[key])
                if not all((expected / name).is_file() for name in ["provenance.json", *[f"{c}.npy" for c in self.channels]]):
                    raise ValueError("Prewarmed view disappeared")
            directory = cache_views(image_path, self.config, self.root, view_names=self.channels)
            if key in self.paths and Path(self.paths[key]) != directory:
                raise ValueError("Prewarmed view provenance changed")
            self.paths[key] = str(directory)
            files = [image_path, self.config, directory / "provenance.json", *[directory / f"{c}.npy" for c in self.channels]]
            self.verified[key] = [(str(path), file_state(path)) for path in files]
        else:
            if any(file_state(path) != state for path, state in self.verified[key]):
                raise ValueError("Verified view/source/config changed during use")
            directory = Path(self.paths[key])
        arrays = {c: np.load(directory / f"{c}.npy", mmap_mode="r", allow_pickle=False) for c in self.channels}
        return arrays

    def provenance(self, image_path):
        return read_json(Path(self.paths[str(Path(image_path))]) / "provenance.json")
