import os
from .interfaces import IStorageBackend

class LocalFileStorage(IStorageBackend):
    """Concrete class to process storage to standard file systems."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save(self, content: str, filename: str) -> None:
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
