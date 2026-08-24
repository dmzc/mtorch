import importlib
from pathlib import Path
import sys


def test_package_import():
    root_dir = Path(__file__).parent.parent.parent
    root_dir_str = str(root_dir)
    if root_dir_str not in sys.path:
        sys.path.insert(0, root_dir_str)

    start_index = len(root_dir.parts) - 1
    errors = []
    for item in root_dir.rglob("*"):  # 所有目录和文件
        parts = item.parts
        if item.is_file() and item.name == "__init__.py" and "tests" not in parts:
            print(parts)
            pkg_name = ".".join(parts[start_index : len(parts) - 1])
            module = None
            import_error = None
            symbol_error = []
            try:
                module = importlib.import_module(pkg_name)
            except Exception as e:
                import_error = repr(e)
            if module is not None:
                if hasattr(module, "__all__"):
                    module_symbols = module.__all__
                    if len(module_symbols) > 0:
                        for item in module_symbols:
                            if not hasattr(module, item):
                                symbol_error.append(item)
            if import_error is not None or len(symbol_error) > 0:
                error = {}
                if import_error is not None:
                    error["import_error"] = import_error
                if len(symbol_error) > 0:
                    error["symbol_error"] = symbol_error
                errors.append({pkg_name: error})
    assert len(errors) == 0, "motorch所有包都能够成功导入且__all__正常"
