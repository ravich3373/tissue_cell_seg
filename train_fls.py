import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2


transform_train = transforms.Compose([
                transforms.ToPILImage(),
                            transforms.ToTensor(),
                                                    ])

#data_root = "/scratch/rc5124/datasets/tissuenet"
#train_data_dict = np.load(os.path.join(data_root, f"tissuenet_v1.1_train.npz"))
#val_data_dict = np.load(os.path.join(data_root, f"tissuenet_v1.1_val.npz"))
#test_data_dict = np.load(os.path.join(data_root, f"tissuenet_v1.1_test.npz"))

#data_dicts = {"train": {"X": train_data_dict["X"], "y": train_data_dict["y"]},
#        "val": {"X": val_data_dict["X"], "y": val_data_dict["y"]},
#        "test": {"X": test_data_dict["X"], "y": test_data_dict["y"]}}

class TissueNetNucleus(Dataset):
    def __init__(self, split, transform=None):
        self.transforms = transform
        fls = glob.glob(f"{split}/*.png")
        self.imgs = sorted([fl for fl in fls if "msk" not in fl])
        self.ann = sorted([fl for fl in fls if "msk" in fl])

    def __getitem__(self, idx):
        dapi_stain_img =  cv2.imread(self.imgs[idx])  #self.imgs[idx][:,:,0]
        instance_mask = cv2.imread(self.ann[idx], 0)  #data_dicts[self.split]["y"][idx][:,:,0]  #self.segs[idx][:,:,0]
        mask = instance_mask > 0
        if self.transforms is  not None:
            dapi_stain_img = self.transforms(dapi_stain_img)
        mask = np.expand_dims(mask, axis=0)
        return dapi_stain_img, mask

    def __len__(self):
        return len(self.imgs)

train_dataset = TissueNetNucleus( "train", transform=transform_train)
valid_dataset = TissueNetNucleus( "val", transform=transform_train)
test_dataset = TissueNetNucleus( "test", transform=transform_train)


print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")


transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
                        ])

transform_val=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)


class PetModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.train_step_ops = []
        self.val_step_ops = []
        self.test_step_ops = []

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        op = self.shared_step(batch, "train")
        self.train_step_ops.append(op)
        return op

    def on_train_epoch_end(self):
        #import pdb; pdb.set_trace()
        print("train epch end")
        op = self.shared_epoch_end(self.train_step_ops, "train")
        self.train_step_ops.clear()  # free mem
        return op

    def validation_step(self, batch, batch_idx):
        op = self.shared_step(batch, "valid")
        self.val_step_ops.append(op)
        return op

    def on_validation_epoch_nd(self):
        print("validation epoch end")
        op = self.shared_epoch_end(self.val_step_ops, "valid")
        self.train_step_ops.clear()
        return op

    def test_step(self, batch, batch_idx):
        op = self.shared_step(batch, "test")
        self.test_step_ops.append(op)
        return op

    def on_test_epoch_end(self):
        op = self.shared_epoch_end(self.test_step_ops, "test")
        self.test_step_ops.clear()
        return op

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

model = PetModel("FPN", "resnet34", in_channels=3, out_classes=1)



trainer = pl.Trainer(
    accelerator="auto", 
    max_epochs=1,
)


trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader,
)

valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
pprint(valid_metrics)

test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
pprint(test_metrics)

