import torch

class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Paths
        self.train_image_dir = "./datasets/VeRi/image_train"
        self.query_image_dir = "./datasets/VeRi/image_query"
        self.gallery_image_dir = "./datasets/VeRi/image_test"
        
        # Model Params
        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_dim = 512
        self.feat_dim = 2048
        self.num_classes = 576  # Will be updated dynamically by dataset
        
        # Training Params
        self.batch_size = 64
        self.num_workers = 4
        self.epochs = 120
        self.learning_rate = 1e-4
        self.weight_decay = 5e-4

cfg = Config()