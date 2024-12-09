from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.exceptions import MissingDependencyException

class LzwCompressor(BaseCompressor):
    
    def __init__(self) -> None:
        super().__init__(self)
        try:
            import lzw
        except ModuleNotFoundError as e:
            raise MissingDependencyException("lzw") from e
        self.compressor = lzw