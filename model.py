import argparse
from collections import OrderedDict
import copy
import itertools
import json
import logging
import os
import sys
import time
from typing import Tuple, Any, Optional, Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import sagemaker_containers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class TransDatasetFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        target_transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        post_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None
    ) -> None:
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=None,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.pre_transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        affine_sample, affine_params = RandomAffineParam(degrees=180, translate=(0, 0.2), scale=(0.8, 1.2), shear=30)(sample)
        sim_sample, sim_params = RandomAffineParam(degrees=180, translate=(0, 0.2), scale=(0.8, 1.2))(sample)
        #euclidean_sample, euclidean_params = RandomAffineParam(degrees=180, translate=(0, 0.2))(sample)
        #ccbs_sample, ccbs_params = ColorJitterParam((0.2,1.8), (0.2,1.8), (0.2,1.8), (-0.2,0.2))(sample)        
        
        sample = self.post_transform(sample)
        affine_sample = self.post_transform(affine_sample)
        sim_sample = self.post_transform(sim_sample)
        #euclidean_sample = self.post_transform(euclidean_sample)
        #ccbs_sample = self.post_transform(ccbs_sample)

        #return sample, affine_sample, affine_params, sim_sample, sim_params, euclidean_sample, euclidean_params, ccbs_sample, ccbs_params, target
        return sample, affine_sample, affine_params, sim_sample, sim_params, target

class RandomAffineParam(transforms.RandomAffine):
    def forward(self, img):
        img_size = torchvision.transforms.functional._get_image_size(img)

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        transformed = torchvision.transforms.functional.affine(img, *ret)
        ret = torch.Tensor([y for x in ret for y in (x if isinstance(x, tuple) else (x,))])
        return transformed, ret

class ColorJitterParam(transforms.ColorJitter):
    def forward(self, img):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = torchvision.transforms.functional.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = torchvision.transforms.functional.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = torchvision.transforms.functional.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = torchvision.transforms.functional.adjust_hue(img, hue_factor)

        return img, torch.Tensor((brightness_factor, contrast_factor, saturation_factor, hue_factor))

def _get_dataloaders(
    input_size, 
    batch_size, 
    val_batch_size, 
    data_dir
):
    pre_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size)                            
    ])
    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.6383, 0.5834, 0.5287], [0.2610, 0.2711, 0.2873])                           
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.6383, 0.5834, 0.5287], [0.2610, 0.2711, 0.2873])                                             
    ])

    logging.info("Get training dataloader")
    trans_dataset = TransDatasetFolder(os.path.join(data_dir, 'training'), pre_transform=pre_transform, post_transform=post_transform)
    train_dataloader = torch.utils.data.DataLoader(trans_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    logging.info("Get validation dataloader")
    validation_dataset = ImageFolder(os.path.join(data_dir, 'validation'), validation_transform)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataloader, validation_dataloader

def _load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2', pretrained=False)

    backbone = nn.Sequential(
        OrderedDict([
            ("conv1", model.conv1),
            ("bn1", model.bn1),
            ("relu", model.relu),
            ("maxpool", model.maxpool),
            ("layer1", model.layer1),
            ("layer2", model.layer2),
            ("layer3", model.layer3)
        ])
    )

    decoders = {
        t: nn.Sequential(
            OrderedDict([
                ("layer4", copy.deepcopy(model.layer4)), 
                ("avgpool", copy.deepcopy(model.avgpool)),
                ("flatten", nn.Flatten()),
                ("fc", copy.deepcopy(model.fc))
            ])
        ) 
        for t in ['affine', 'similarity']
    }

    classifier = nn.Sequential(
        OrderedDict([
            ("layer4", copy.deepcopy(model.layer4)), 
            ("avgpool", copy.deepcopy(model.avgpool)),
            ("flatten", nn.Flatten()),
            ("fc", nn.Linear(2048, 12))
        ])
    ) 

    decoders['affine'].fc = nn.Linear(2048, 6)
    decoders['similarity'].fc = nn.Linear(2048, 6)
    #decoders['euclidean'].fc = nn.Linear(2048, 6)
    #decoders['ccbs'].fc = nn.Linear(2048, 4)

    return backbone, decoders, classifier

def train(args):
    logger.debug(f"Number of gpus available - {args.num_gpus}")
    device = torch.device('cuda' if args.num_gpus > 0 else 'cpu')

    train_dataloader, validation_dataloader = _get_dataloaders(args.input_size, args.batch_size, args.val_batch_size, args.data_dir)

    backbone, decoders, classifier = _load_model()
    backbone = backbone.to(device)
    decoders['affine'] = decoders['affine'].to(device)
    decoders['similarity'] = decoders['similarity'].to(device)
    classifier = classifier.to(device)

    backbone_params = [backbone.parameters(), classifier.parameters()]
    backbone_optimizer = optim.Adam(itertools.chain(*backbone_params), lr=args.bacbone_lr)
    decoder_params = [d.parameters() for _, d in decoders.items()]
    aet_optimizer = optim.SGD(itertools.chain(*decoder_params), lr=args.aet_lr)
    criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    criterion = criterion.to(device)
    class_criterion = class_criterion.to(device)
    kl_criterion = kl_criterion.to(device)

    since = time.time()
    it = 0
    affine_loss_weight = 0
    affine_loss_inc = 0.75/16000
    sim_loss_weight = 0
    sim_loss_inc = 0.5/16000
    kl_loss_weight = 0
    kl_loss_inc = 1/16000

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{ args.epochs }")
        print('-' * 10)

        backbone.train()
        for _, d in decoders.items():
            d.train()
        classifier.train()
        running_loss = 0.0
        running_corrects = 0

        #for original, affine_sample, affine_params, sim_sample, sim_params, euclidean_sample, euclidean_params, ccbs_sample, ccbs_params, target in dataloader:
        for original, affine_sample, affine_params, sim_sample, sim_params, target in train_dataloader:
            backbone_optimizer.zero_grad()
            aet_optimizer.zero_grad()
            original = original.to(device)
            affine_sample = affine_sample.to(device)
            affine_params = affine_params.to(device)
            sim_sample = sim_sample.to(device)
            sim_params = sim_params.to(device)
            target = target.to(device)

            # Get encodings from backbone
            encoded = backbone(original)
            encoded_affine = backbone(affine_sample)
            encoded_similarity = backbone(sim_sample)
            #encoded_euclidean = backbone(euclidean_sample)
            #encoded_ccbs = backbone(ccbs_sample)

            # Get decodings for transformation params
            affine_decoded = decoders['affine'](encoded_affine)
            similarity_decoded = decoders['similarity'](encoded_similarity)
            #euclidean_decoded = decoders['euclidean'](encoded_euclidean)
            #ccbs_decoded = decoders['ccbs'](encoded_ccbs)

            # Get class prob output and log-prob for KLLoss
            outputs = classifier(encoded)
            outputs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            trans_affine_output = classifier(encoded_affine)
            trans_affine_output = F.log_softmax(trans_affine_output, dim=1)
            trans_sim_output = classifier(encoded_similarity)
            trans_sim_output = F.log_softmax(trans_sim_output, dim=1)
            
            affine_loss_weight = min(0.75, affine_loss_weight + affine_loss_inc)
            sim_loss_weight = min(0.5, sim_loss_weight + sim_loss_inc)
            kl_loss_weight = min(1, kl_loss_weight + kl_loss_inc)

            # Calculate weighted loss
            loss = (
                affine_loss_weight*criterion(affine_decoded, affine_params) + 
                sim_loss_weight*criterion(similarity_decoded, sim_params) +
           #     0.2*criterion(euclidean_decoded, euclidean_params) +
            #    0.05*criterion(ccbs_decoded, ccbs_params) +
                kl_loss_weight * (
                    kl_criterion(trans_affine_output, outputs) + 
                    kl_criterion(trans_sim_output, outputs)
                ) +
                class_criterion(outputs, target)
            )

            loss.backward()
            backbone_optimizer.step()
            aet_optimizer.step()

            running_loss += loss.item() * original.size(0)
            running_corrects += torch.sum(preds == target.data)
            it += 1

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset)
        logger.info(f"Training Loss: {epoch_loss} Training Acc: {epoch_acc}")
        
        backbone.eval()
        for _, d in decoders.items():
            d.eval()
        classifier.eval()

        val_loss = 0.0
        running_corrects = 0
        for original, target in validation_dataloader:
            original = original.to(device)
            target = target.to(device)
            encoded = backbone(original)
            outputs = classifier(encoded)
            _, preds = torch.max(outputs, 1)
            loss = class_criterion(outputs, target)
            running_corrects += torch.sum(preds == target.data)
            val_loss += loss.item() * original.size(0)
        
        val_acc = running_corrects.double() / len(validation_dataloader.dataset)
        val_loss = val_loss / len(validation_dataloader.dataset)

        logger.info(f"Val Loss: {val_loss} Val Acc: {val_acc}")
    
    time_elapsed = time.time() - since
    logger.info(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")
    final_model = nn.Sequential(backbone, classifier)

    save_model(final_model, args.model_dir)

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
        '--val-batch-size', 
        type=int, 
        default=1000,
        help='input batch size for validation (default: 1000)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help='number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--backbone-lr', 
        type=float,
        default=0.002,
        help='backbone learning rate (default: 0.002)'
    )
    parser.add_argument(
        '--aet-lr', 
        type=float, 
        default=0.1,
        help='aet learning rate (default: 0.1)'
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