import os
from lib.data_setup2 import *
from lib.classifier import *
from lib.generator import *
from lib.discriminator import *
from lib.hierarchical_model import *
from lightning.fabric.strategies.fsdp import FSDPStrategy

# logging utilities
from lightning.fabric.loggers import TensorBoardLogger
from torchvision.utils import make_grid

train = TensorMRIDatasetLazy(
    train_paths, train_labels, lambda x, _: transforms_affine(mri_transform(x, _))
)
train_loader = DataLoader(train, 2, shuffle=True)


# util functions
def perturb_image(imgs, prob):
    if random.random() > prob:
        return imgs
    d2 = torch.randn(
        imgs.shape[0], *[i // 8 for i in imgs.shape[2:]], 3
    ) / random.randint(45, 100)
    d2 = fabric.to_device(d2)
    d2 = nn.functional.interpolate(
        d2.permute(0, 4, 1, 2, 3), scale_factor=8, mode="trilinear"
    ).permute(0, 2, 3, 4, 1)
    d2 = d2 + id_mat
    return nn.functional.grid_sample(
        imgs, d2.permute(0, 3, 2, 1, 4), align_corners=False
    )


# setting training evironment
logger = TensorBoardLogger(root_dir="./counterfactual_f", default_hp_metric=False)

if "SC_NODES" in os.environ:
    print(f"Nodes: {os.environ["SC_NODES"]}")
    fabric = L.Fabric(
        strategy=FSDPStrategy(accelerator="gpu", sharding_strategy="NO_SHARD"),
        num_nodes=int(os.environ["SC_NODES"]),
        loggers=logger,
    )
else:
    fabric = L.Fabric(
        strategy=FSDPStrategy(accelerator="gpu", sharding_strategy="NO_SHARD"),
        loggers=logger,
    )
    # debug mode
    # fabric = L.Fabric(accelerator="cpu", loggers=logger)
fabric.launch()


# load pre-trained classifier
model = torch.load("./PATH/TO/classifier1.ckpt", map_location="cpu")
classifier = Classifier1(norm=lambda x: nn.BatchNorm3d(x))
classifier.load_state_dict(model["classifier"])


class VoxelDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet(2, 1, features=[32, 64, 128, 256])

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


# instantiate GAN modules


def foo(scale):
    if scale == 1:
        return DiffeomorphicTransform(
            2, 3, features=[16, 32, 64, 128], size=128 // scale
        )
    else:
        return DiffeomorphicTransform(2, 3, features=[16, 32, 64], size=128 // scale)


transform = HierarchicalTransform(backbone=foo)
disc = PatchDiscriminator()

# hyperparameters
LR = 1e-4
WEIGHT_DECAY = 1e-3
LOSS_CLS = 1
LOSS_RECON = 1
LOSS_SMOOTH = 1
# LOSS_INVERSE_CONSISTENCY = 1
LOSS_OCC = 1
LOSS_GAUSS = 1

opt_g = torch.optim.Adam(
    transform.parameters(),
    LR,
    weight_decay=WEIGHT_DECAY,
)

opt_d = torch.optim.Adam(
    disc.parameters(),
    LR,
    weight_decay=WEIGHT_DECAY,
)

# no optimization step performed but required for generating gradients for classifier weights
opt_c = torch.optim.Adam(classifier.parameters(), LR)

# moving models and data loaders to fabric
transform, opt_g = fabric.setup(transform, opt_g)
disc, opt_d = fabric.setup(disc, opt_d)
classifier, opt_c = fabric.setup(classifier, opt_c)
train_loader = fabric.setup_dataloaders(train_loader)

classifier.eval()
transform.train()
disc.train()

# identity mapping volume coordinates for spatial transformer network layer
id_mat = identity_deform_field_3D([128, 128, 128])
id_mat = fabric.to_device(id_mat)

step_count = 0
for epoch in range(500):
    print(f"Epoch: {epoch+1}")
    disc_trick_counts = [0, 0]  # tricked count, total images
    class_trick_counts = [0, 0]

    # batch iteration
    for b_idx, batch in enumerate(train_loader):
        opt_g.zero_grad()
        imgs, targets = batch
        imgs2, _ = next(iter(train_loader))  # real images for discriminator

        if imgs2.shape[0] != imgs.shape[0]:  # same shape for concatenation
            continue

        with torch.inference_mode():
            actual = classifier(imgs)[0]
            actual = (actual.sigmoid() > 0.5).float()

        # randomly chosen class label to push image to
        to_domain_labels = fabric.to_device(
            torch.randint(0, 2, (imgs.shape[0],))
        ).type_as(imgs)

        target_label = fabric.to_device(torch.ones_like(imgs)) * fabric.to_device(
            to_domain_labels.view(imgs.shape[0], 1, 1, 1, 1)
        )

        orig_label = fabric.to_device(torch.ones_like(imgs)) * fabric.to_device(
            (torch.tensor(actual)).view(imgs.shape[0], 1, 1, 1, 1)
        )

        # generator training
        # ==================
        # original to target domain
        imgs_t, t = transform(imgs, target_label)
        t_disc = disc(imgs_t, imgs2)
        t_class = classifier(imgs_t)[0]
        t_preds = (t_class.detach().sigmoid() > 0.5).float()

        g_loss_adv = nn.functional.binary_cross_entropy_with_logits(
            t_disc, fabric.to_device(torch.ones_like(t_disc).type_as(t_disc))
        )

        # print(t_class, to_domain_labels)
        g_loss_cls = nn.functional.binary_cross_entropy_with_logits(
            t_class.reshape(-1),
            to_domain_labels.reshape(-1),
        )

        # target to original domain
        imgs_recon, _ = transform(imgs_t, orig_label)  # added detach
        g_loss_smooth = smooth_loss_l2(t)  # smoothing disp
        g_loss_norm = (t**2).mean()
        gauss_loss = torch.mean(
            torch.abs(
                nn.functional.conv3d(
                    imgs_t,
                    fabric.to_device(torch.ones(1, 1, 5, 5, 5)) / 5**3,
                    padding=0,
                )
                - nn.functional.conv3d(
                    imgs,
                    fabric.to_device(torch.ones(1, 1, 5, 5, 5)) / 5**3,
                    padding=0,
                )
            )
        )
        loss_gen = (
            g_loss_adv
            + g_loss_cls * LOSS_CLS
            + g_loss_adv * g_loss_cls * g_loss_smooth * gauss_loss * g_loss_norm
            + g_loss_smooth * LOSS_SMOOTH
            + gauss_loss * LOSS_GAUSS
            + g_loss_norm * 1e-3
        )

        fabric.backward(loss_gen)

        opt_g.step()

        class_trick_counts[0] += torch.sum(
            (t_class.sigmoid().detach() > 0.5).type_as(t_class) == to_domain_labels
        ).item()
        class_trick_counts[1] += len(to_domain_labels)

        disc_trick_counts[0] += torch.sum(
            (t_disc.sigmoid() > 0.5) == fabric.to_device(torch.ones_like(t_disc))
        )
        disc_trick_counts[1] += len(t_disc.reshape(-1))

        # train discriminator
        # ===================
        torch.cuda.empty_cache()
        mod_imgs = imgs
        mod_imgs2 = imgs2

        real_pred = disc(mod_imgs, mod_imgs2)
        fake_pred = disc(imgs_t.detach(), mod_imgs)

        d_loss_real = nn.functional.binary_cross_entropy_with_logits(
            real_pred,
            fabric.to_device(torch.ones_like(real_pred).type_as(real_pred)),
        )

        d_loss_fake = nn.functional.binary_cross_entropy_with_logits(
            fake_pred,
            fabric.to_device(torch.zeros_like(fake_pred).type_as(fake_pred)),
        )

        loss_disc = d_loss_real + d_loss_fake

        fabric.backward(loss_disc)
        opt_d.step()
        opt_d.zero_grad()

        # logging
        # =======
        if b_idx % 2 == 0:
            fabric.log_dict(
                {
                    "g_loss_adv": g_loss_adv,
                    "g_loss_cls": g_loss_cls,
                    "g_loss_norm": g_loss_norm,
                    "g_loss_smooth": g_loss_smooth,
                    "loss_gen": loss_gen,
                    "d_loss_real": d_loss_real,
                    "d_loss_fake": d_loss_fake,
                    "loss_disc": loss_disc,
                    "domain_transfer_acc_running": class_trick_counts[0]
                    / class_trick_counts[1],
                    "disc_trick_acc_running": disc_trick_counts[0]
                    / disc_trick_counts[1],
                },
                step=step_count,
            )

            s = 50
            image = make_grid(
                torch.cat(
                    [
                        (imgs)[:1, :, :, :, s],
                        (imgs_t)[:1, :, :, :, s],
                        torch.abs((imgs)[:1, :, :, :, s] - (imgs_t)[:1, :, :, :, s]),
                        (imgs)[:1, :, :, :, s + 10],
                        (imgs_t)[:1, :, :, :, s + 10],
                        torch.abs(
                            (imgs)[:1, :, :, :, s + 10] - (imgs_t)[:1, :, :, :, s + 10]
                        ),
                    ]
                ),
                nrow=3,
            )
            fabric.logger.experiment.add_image("Gen_images", image, step_count)

            fabric.logger.experiment.add_image(
                "Move_images",
                make_grid(torch.cat([(t[0].squeeze() / t[0].max())[:, :, :, s]])),
                step_count,
            )
            step_count += 1

    fabric.log_dict(
        {
            "domain_transfer_acc": class_trick_counts[0] / class_trick_counts[1],
            "disc_trick_acc": disc_trick_counts[0] / disc_trick_counts[1],
        },
        step=step_count,
    )

    torch.save(
        {
            "transform": transform.state_dict(),
            "disc": disc.state_dict(),
            "spec": "Hierarchical Backbone model [32, 64, 128, 256], l2 smoothing",
        },
        "./generative_model.ckpt",
    )

print("Fin.")
