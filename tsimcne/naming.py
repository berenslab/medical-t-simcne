from torchvision import transforms

augmentation_mapping = {
    'RandomApply': 'functrot', 
    'RandomRotation': 'randrot',
    'RandomResizedCrop': 'RC',
    'RandomHorizontalFlip': 'HF',
    'RandomVerticalFlip': 'VF',
    'ColorJitter': 'JI',
    'RandomGrayscale': 'GR',
    'ToTensor': '' 
}

def aug2string(transforms_compose):
    short_strings = []
    for transform in transforms_compose.transforms:
        transform_name = type(transform).__name__
        if transform_name == 'RandomApply':
            for t in transform.transforms:
                if isinstance(t, transforms.ColorJitter):
                    short_strings.append(augmentation_mapping['ColorJitter'])
                else:
                    short_strings.append(augmentation_mapping.get(transform_name, 'unknown'))
        else:
            short_strings.append(augmentation_mapping.get(transform_name, 'unknown'))
    
    final_string = '_'.join(filter(None, short_strings))
    print('final string \n', final_string)
    return final_string