import torchvision.transforms as T

def get_train_transforms():
    """Augmentations for robust training"""
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.Pad(10),
        T.RandomCrop(256, 256),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def get_inference_transforms():
    
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

train_transforms = get_train_transforms()
inference_transforms = get_inference_transforms()