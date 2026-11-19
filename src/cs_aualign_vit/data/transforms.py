"""Online transforms are separate from native-resolution offline crops."""


def build_transform(
    training, image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
):
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    if image_size <= 0 or len(mean) != 3 or len(std) != 3 or any(x <= 0 for x in std):
        raise ValueError("Invalid image size or RGB normalization")
    operations = [T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)]
    if training:
        operations.append(T.RandomHorizontalFlip(p=0.5))
    operations.extend([T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose(operations)


class GuidanceTransform:
    """Apply one random flip decision to an image and its 7x7 AU mask."""

    def __init__(self, image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode

        self.resize = T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean, std)

    def __call__(self, image, mask):
        import torch
        from torchvision.transforms import functional as TF

        image = self.resize(image)
        if torch.rand(()) < 0.5:
            image = TF.hflip(image)
            mask = mask.reshape(7, 7, -1).flip(1).reshape(49, -1)
        return self.normalize(self.to_tensor(image)), mask


class WeakStrongTransform:
    """USTST target views: ``X_Ts`` is ``X_Tw`` after RandAugment (Section 3).

    The weak view uses the POST training transform (resize + horizontal flip);
    the strong view applies RandAugment to that same flipped image, so the two
    views differ only by the photometric/geometric RandAugment operations.
    """

    def __init__(
        self,
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        *,
        num_ops=2,
        magnitude=9,
    ):
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode

        if image_size <= 0 or len(mean) != 3 or len(std) != 3 or any(x <= 0 for x in std):
            raise ValueError("Invalid image size or RGB normalization")
        self.geometry = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.rand_augment = T.RandAugment(num_ops=num_ops, magnitude=magnitude)
        self.finish = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def __call__(self, image):
        weak = self.geometry(image)
        return self.finish(weak), self.finish(self.rand_augment(weak))
