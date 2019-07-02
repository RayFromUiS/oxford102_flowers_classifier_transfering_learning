
from torchvision import datasets, transforms
import torch

def preprocess_data(data_dir):

    # Data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define  transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(10),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = {}
    image_datasets["train"] = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_datasets["valid"] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    image_datasets["test"] = datasets.ImageFolder(test_dir, transform=test_transforms)

    #  Define the dataloaders
    train_loader = torch.utils.data.DataLoader(image_datasets["train"], batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(image_datasets["valid"], batch_size=32)
    test_loader = torch.utils.data.DataLoader(image_datasets["test"], batch_size=32)

    print(f"Data loaded finished from {data_dir} directory.")

    return image_datasets, train_loader, valid_loader, test_loader


