from torchvision import transforms

def oct_data_transforms(cfg):
    data_aug = cfg.data.data_augmentation
    aug_args = cfg.oct_data_augmentation_args

    operations = {
        'random_crop': random_apply(
            transforms.RandomResizedCrop(
                size=(cfg.data.input_size, cfg.data.input_size),
                scale=aug_args.random_crop.scale,
                ratio=aug_args.random_crop.ratio
            ),
            p=aug_args.random_crop.prob
        ),
        'horizontal_flip': transforms.RandomHorizontalFlip(
            p=aug_args.horizontal_flip.prob
        ),
        'vertical_flip': transforms.RandomVerticalFlip(
            p=aug_args.vertical_flip.prob
        ),
        'color_distortion': random_apply(
            transforms.ColorJitter(
                brightness=aug_args.color_distortion.brightness,
                contrast=aug_args.color_distortion.contrast,
                saturation=aug_args.color_distortion.saturation,
                hue=aug_args.color_distortion.hue
            ),
            p=aug_args.color_distortion.prob
        ),
        'rotation': random_apply(
            transforms.RandomRotation(
                degrees=aug_args.rotation.degrees,
                fill=aug_args.value_fill
            ),
            p=aug_args.rotation.prob
        ),
        'translation': random_apply(
            transforms.RandomAffine(
                degrees=0,
                translate=aug_args.translation.range,
                fill=aug_args.value_fill
            ),
            p=aug_args.translation.prob
        )
    }

    augmentations = []
    for op in data_aug:
        if op not in operations:
            raise NotImplementedError('Not implemented data augmentation operations: {}'.format(op))
        augmentations.append(operations[op])

    normalization = [
        transforms.Resize((cfg.data.input_size)), # (cfg.data.input_size, cfg.data.input_size)
        transforms.ToTensor(),
        transforms.CenterCrop(cfg.data.input_size),
        transforms.Normalize(cfg.data.mean, cfg.data.std)
    ]

    train_preprocess = transforms.Compose([
        *augmentations,
        *normalization
    ])

    if cfg.data.push_with_augmentation:
        train_push_preprocess = transforms.Compose([
        *augmentations,
        *normalization[:-1] ])
        print('push with augmentation')
    else:
        train_push_preprocess = transforms.Compose([*normalization[:-1] ])
        print('push without augmentation')

    test_preprocess = transforms.Compose(normalization)

    return train_preprocess, train_push_preprocess, test_preprocess

def random_apply(op, p):
    return transforms.RandomApply([op], p=p)
