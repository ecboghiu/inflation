def blank_tqdm(*args, **kwargs):
    try:
        if not kwargs['disable']:
            print(kwargs['desc'])
    except KeyError:
        pass
    return args[0]
