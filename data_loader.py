from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from config import data_path, transform_train, transform_val, batch_size, num_worker

def get_data_loaders(full_dataset, train_ratio, batch_size=batch_size, num_workers=num_worker):
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_dataset.dataset.transform = transform_train
    test_dataset.dataset.transform = transform_val
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader

def load_dataset():
    full_dataset = ImageFolder(data_path, transform=None)
    return full_dataset