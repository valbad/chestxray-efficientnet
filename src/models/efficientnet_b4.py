import timm


def create_model(num_classes=3, pretrained=True):
    """
    Load an EfficientNet-B4 model from timm and set the classifier
    to the desired number of classes.
    """
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate = 0.4,
        drop_path_rate = 0.2, 
        in_chans = 1
    )
    return model
