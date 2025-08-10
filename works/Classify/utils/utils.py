def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst