import argparse
import copy
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def topk_correct(output, target, k=3):
    with torch.no_grad():
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)

        return correct_k.item()

def _get_dataloaders(
    input_size, 
    batch_size,
    data_dir
):
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.6383, 0.5834, 0.5287], [0.2610, 0.2711, 0.2873])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.6383, 0.5834, 0.5287], [0.2610, 0.2711, 0.2873])
        ]),
    }

    logger.info("Get dataloaders")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['training', 'validation']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['training', 'validation']}

    return dataloaders_dict

def _load_model():
    model_ft = models.resnext50_32x4d(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = True
    n_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(n_features, 12)
    return model_ft

def train(args):
    logger.debug(f"Number of gpus available - {args.num_gpus}")
    device = torch.device('cuda' if args.num_gpus > 0 else 'cpu')

    dataloaders = _get_dataloaders(args.input_size, args.batch_size, args.data_dir)

    model = _load_model()
    model = model.to(device)
    params_to_update = model.parameters()
    optimizer = optim.SGD(params_to_update, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    since = time.time()
    best_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{ args.epochs }")
        print('-' * 10)

        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            running_topk_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_topk_corrects += topk_correct(outputs, labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_topk = running_topk_corrects / len(dataloaders[phase].dataset)

            logger.info(f"{phase} Loss: {epoch_loss} {phase} Acc: {epoch_acc} {phase} Top 3 Acc: {epoch_topk}")
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
        print()
    
    time_elapsed = time.time() - since
    logger.info(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")
    print(f"Best validation Acc: {best_acc}")

    model.load_state_dict(best_model_weights)
    save_model(model, args.model_dir)

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        '--input-size',
        type=int,
        default=224,
        help="size of input images"
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=64,
        help='input batch size for training (default: 64)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help='number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--lr', 
        type=float,
        default=1e-3,
        help='backbone learning rate (default: 0.002)'
    )

    # Container environment
    parser.add_argument(
        '--hosts', 
        type=list, 
        default=json.loads(os.environ['SM_HOSTS'])
    )
    parser.add_argument(
        '--current-host', 
        type=str, 
        default=os.environ['SM_CURRENT_HOST']
    )
    parser.add_argument(
        '--model-dir', 
        type=str, 
        default=os.environ['SM_MODEL_DIR']
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default=os.environ['SM_CHANNEL_TRAINING']
    )
    parser.add_argument(
        '--num-gpus', 
        type=int, 
        default=os.environ['SM_NUM_GPUS']
    )

    train(parser.parse_args())