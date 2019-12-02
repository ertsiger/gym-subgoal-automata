def get_param(params, param_name, default_value=None):
    if params is not None and param_name in params:
        return params[param_name]
    return default_value
