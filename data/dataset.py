import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class CarlaVeriDataset(Dataset):
    """
    Standard Re-ID Dataset parser.
    Expects filenames in VeRi format: {PID}_c{CAMID}_{FRAME}_{SEQ}.jpg
    Example: 0002_c002_00030600_0.jpg
    """
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        
        # Grab all images
        self.image_paths = glob.glob(os.path.join(self.dir_path, "*.jpg"))
        
        self.data = []
        pids = set()
        
        for path in self.image_paths:
            filename = os.path.basename(path)
            parts = filename.split('_')
            
            # Extract PID and CamID based on standard VeRi naming
            pid = parts[0]
            camid = int(parts[1][1:]) # e.g., 'c002' -> 2
            
            # Skip junk images (-1)
            if pid == '-1': 
                continue
                
            self.data.append((path, pid, camid))
            pids.add(pid)
            
        # Map original string PIDs to contiguous integers [0, N-1]
        self.unique_pids = sorted(list(pids))
        self.pid2label = {pid: label for label, pid in enumerate(self.unique_pids)}
        
        # Precompute the mapped labels for the PKSampler
        self.new_pids = [self.pid2label[pid] for _, pid, _ in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, original_pid, camid = self.data[idx]
        label = self.pid2label[original_pid]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        # Returning 5 items: image, mapped_label, camid, img_path, original_pid
        return img, label, camid, img_path, original_pid