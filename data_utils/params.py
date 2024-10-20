from ruamel.yaml import YAML


class Params:
    '''Inspired by:
    https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams'''
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            # Load and find
            yaml = YAML(typ='safe')
            self.yaml_map = yaml.load(f)