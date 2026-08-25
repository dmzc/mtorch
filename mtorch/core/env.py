def pkg_enabled(pkg_name: str, pkg_version: str = None) -> bool:
    r"""
    判断环境是否安装了某个包。

    Args:
        pkg_name:包名。
        pkg_version:包版本，不传递则校验版本。
    """
    from importlib.util import find_spec

    if find_spec(pkg_name) is None:
        return False
    if pkg_version is None:
        return True

    from importlib.metadata import version

    return version(pkg_name) == pkg_version


# 是否启用反向传播
ENABLE_BACKPROGATION = True  # TODO:不用这个，应该是自动判断的


import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent
