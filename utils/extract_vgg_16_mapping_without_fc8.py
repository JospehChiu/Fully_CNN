def extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping):
    """Removes the fc8 variable mapping from FCN-32s to VGG-16 model mapping dict.
    Given the FCN-32s to VGG-16 model mapping dict which is returned by FCN_32s()
    function, remove the mapping for the fc8 variable. This is done because this
    variable is responsible for final class prediction and is different for different
    tasks. Last layer usually has different size, depending on the number of classes
    to be predicted. This is why we omit it from the dict and those variables will
    be randomly initialized later.
    
    Parameters
    ----------
    vgg_16_variables_mapping : dict {string: variable}
        Dict which maps the FCN-32s model's variables to VGG-16 checkpoint variables
        names. Look at FCN-32s() function for more details.
    
    Returns
    -------
    updated_mapping : dict {string: variable}
        Dict which maps the FCN-32s model's variables to VGG-16 checkpoint variables
        names without fc8 layer mapping.
    """
    
    # TODO: review this part one more time
    vgg_16_keys = vgg_16_variables_mapping.keys()

    vgg_16_without_fc8_keys = []

    for key in vgg_16_keys:

        if 'fc8' not in key:
            vgg_16_without_fc8_keys.append(key)

    updated_mapping = {key: vgg_16_variables_mapping[key] for key in vgg_16_without_fc8_keys}
    
    return updated_mapping