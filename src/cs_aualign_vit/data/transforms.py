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
