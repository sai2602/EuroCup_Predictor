def Model_Selector(model_info_path):

    handler = open(model_info_path, 'r')
    model_info = handler.readlines()
    details = model_info[0].split(",")
    return details