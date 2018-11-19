


def nested_iter(d):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            for ki, vi in nested_iter(v):
                yield k + '|' + ki, vi
        else:
            yield k, v

def config_val_string(config):
    config_items = [kv for kv in nested_iter(config)]
    config_vals = map(lambda x: str(x[1]), config_items)
    return ','.join(config_vals)

def config_key_string(config):
    config_items = [kv for kv in nested_iter(config)]
    config_keys = map(lambda x: str(x[0]), config_items)
    return ','.join(config_keys)

