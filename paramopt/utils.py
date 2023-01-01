import os.path as osp


def unique_path(path):
    ret = path
    base, ext = osp.splitext(path)
    i = 1
    while osp.isfile(ret):
        ret = base + f' ({i})' + ext
        i += 1
    return ret
