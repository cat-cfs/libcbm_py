import json


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def load_raw(path):
    '''
    loads configuration json from an existing file
    '''
    with open(path, 'r') as configfile:
        configString = configfile.read()
    return configString


def to_string(config):
    return json.dumps(config, indent=4)#, ensure_ascii=True)


def save_config(path, config):
    with open(path, 'w') as configfile:
        configfile.write(to_string(config))


def initialize_config(save_path=None, **kwargs):
    '''
    initialize config sets up the json configuration object passed to the
    underlying dll returns config as string, and optionally saves to specified
    path
    '''
    configuration = kwargs
    if save_path:
        save_config(save_path, configuration)

    return to_string(configuration)
