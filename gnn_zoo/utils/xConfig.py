from gnn_zoo.utils import io


def load_from_yaml(file_config):
    config = io.load_yaml(file_config)
    return config


def load_from_python_script(file_py):
    raise NotImplementedError


def parse_cmd_args(args, config_to_be_updated=None):
    if config_to_be_updated is None:
        config = {}
    else:
        config = config_to_be_updated
    
    for arg in args:
        field_path, value = _parse_cmd_arg(arg)
        _update_value(config, field_path, value)
    return config


def save_as_yaml(file, config):
    io.save_yaml(file, config)


def _parse_cmd_arg(arg):
    # arg = 'model|user_tower|lr::float::0.01'
    # arg = 'model|user_tower|lr::0.01'
    s = arg.split('::')
    if len(s) == 3:
        s_field_path, s_value_type, s_value = s
        value = eval(s_value_type)(s_value)
    elif len(s) == 2:
        s_field_path, s_value = s
        value = eval(s_value)
    else:
        import pdb; pdb.set_trace()
    field_path = s_field_path.split('|')
    return field_path, value


def _update_value(config, field_path, value):
    sub_config = config
    for field in field_path[:-1]:
        if field not in sub_config:
            sub_config[field] = {}
        sub_config = sub_config[field]
    sub_config[field_path[-1]] = value
