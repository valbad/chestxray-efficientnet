import timm


def create_model(model_type = "efficientnet_b4", num_classes=3, pretrained=True):
    """
    Load an EfficientNet model from timm and set the classifier
    to the desired number of classes.
    """
    model = timm.create_model(
        model_type,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate = 0.4,
        drop_path_rate = 0.2, 
        in_chans = 1
    )
    return model
