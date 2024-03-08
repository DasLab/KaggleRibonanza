from typing import Union
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from collections import namedtuple
from os import path
import numpy as np
import h5py
from pathlib import Path

Feature = Union[str, np.ndarray]
VersionedFeature = namedtuple('VersionedFeature', ['value', 'version'])
FeatureCacheKey = namedtuple('FeatureCacheKey', ['feature', 'sequence'])

class FeatureCache(AbstractContextManager, ABC):
    def __enter__(self):
        global cache
        if cache is not default_cache:
            raise 'A feature cache has already been initialized'
        cache = self

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        global cache
        cache = default_cache

        return None
    
    @abstractmethod
    def get(self, feature: str, sequence: str, version: int = 1) -> Union[Feature, None]:
        raise NotImplementedError()
    
    @abstractmethod
    def set(self, feature: str, sequence: str, value: Feature, version: int = 1):
        raise NotImplementedError()
    
    @abstractmethod
    def exists(self, feature: str, sequence: str):
        raise NotImplementedError()

class NullFeatureCache(FeatureCache):
    def get(self, feature: str, sequence: str, version: int = 1) -> Union[Feature, None]:
        return None

    def set(self, feature: str, sequence: str, value: Feature, version: int = 1):
        pass

    def exists(self, feature: str, sequence: str):
        raise NotImplementedError('You should not be using exists with a null feature cache. This probably means you are precomputing a value, but since it is not cached this will be doing extra work')

class MemoryFeatureCache(FeatureCache):
    def __init__(self):
        self.data: dict[FeatureCacheKey, VersionedFeature] = {}

    def get(self, feature: str, sequence: str, version: int = 1) -> Union[Feature, None]:
        res = self.data.get((feature, sequence))
        
        if not res: return None
        if res.version != version: return None

        return res.value

    def set(self, feature: str, sequence: str, value: Feature, version: int = 1):
        self.data[(feature, sequence)] = VersionedFeature(value, version)

    def exists(self, feature: str, sequence: str):
        return (feature, sequence) in self.data

class FSFeatureCache(FeatureCache):
    def __init__(self, path, gzip_level=4):
        self.path = path
        self.gzip_level = gzip_level

    def __enter__(self):
        # For some reason using the default (recommended) driver, I'm getting errors
        # when existing out of one FSFeatureCache and entering another, so we'll
        # use the python file driver instead.
        fpath = Path(path.join(self.path, 'feature-cache.h5'))
        fpath.touch(exist_ok=True)
        self.f = h5py.File(open(fpath, 'rb+'), 'a')

        return super().__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.f.close()

        return super().__exit__(exc_type, exc_value, exc_tb)

    def get(self, feature: str, sequence: str, version: int = 1, retry = 0) -> Union[Feature, None]:
        res = self.f.get(f'{feature}-{sequence}')

        if not res: return None
        if res.attrs['version'] != version: return None
        
        val = res[()]
        if isinstance(val, bytes): return val.decode()
        
        return val

    def set(self, feature: str, sequence: str, value: Feature, version: int = 1):
        key = f'{feature}-{sequence}'

        if key in self.f:
            del self.f[key]
        
        if isinstance(value, str):
            # Unfortunately scalar datasets don't support compression
            self.f.create_dataset(key, data=value)
        else:
            self.f.create_dataset(key, data=value, compression="gzip", compression_opts=self.gzip_level)
        
        self.f[key].attrs['version'] = version

    def exists(self, feature: str, sequence: str):
        return f'{feature}-{sequence}' in self.f

default_cache = NullFeatureCache()
cache: Union[FeatureCache, None] = default_cache

def get_feature_cache():
    return cache
