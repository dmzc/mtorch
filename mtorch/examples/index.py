import tkinter as tk
from tkinter import ttk
from pathlib import Path
import subprocess
import ast

BASE_DIR = Path(__file__).parent


def get_script_doc(path: Path) -> str:
    """读取脚本模块docstring，作为demo简介"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)
        doc = ast.get_docstring(tree)
        if doc:
            lines = [l.strip() for l in doc.splitlines() if l.strip()]
            return lines[0] if lines else "(无说明)"
        return "(无说明)"
    except Exception:
        return "(读取失败)"


def scan_demos():
    demos = []
    for p in BASE_DIR.rglob("*_demo.py"):
        if p.name.startswith("_") or p.name == "index.py":
            continue
        rel = p.relative_to(BASE_DIR)
        demos.append({"rel": rel, "full": p, "doc": get_script_doc(p)})
    demos.sort(key=lambda d: str(d["rel"]))
    return demos


class DemoLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Demo 启动器 — examples/index")
        self.root.geometry("900x600")

        main_frame = ttk.Frame(root, padding=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_bar = ttk.Frame(main_frame)
        top_bar.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(top_bar, text="Demo列表（双击运行 / 选中点按钮）").pack(side=tk.LEFT)
        ttk.Button(top_bar, text="刷新列表", command=self.reload).pack(side=tk.RIGHT)

        # 树形表格
        columns = ("rel_path", "desc")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="tree headings")
        self.tree.heading("#0", text="目录")
        self.tree.heading("rel_path", text="文件")
        self.tree.heading("desc", text="简介")
        self.tree.column("#0", width=220)
        self.tree.column("rel_path", width=320)
        self.tree.column("desc", width=320)

        vsb = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(8, 0))
        self.run_btn = ttk.Button(
            btn_frame, text="运行选中Demo", command=self.run_selected
        )
        self.run_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(main_frame, textvariable=self.status_var).pack(fill=tk.X, pady=(6, 0))

        self.tree.bind("<Double-1>", self.on_double_click)
        self.demo_list = []
        self.reload()

    def _clear_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def reload(self):
        self._clear_tree()
        self.demo_list = scan_demos()
        folder_nodes = {}

        for d in self.demo_list:
            rel = d["rel"]
            parts = list(rel.parts)
            parent_id = ""
            for idx, part in enumerate(parts[:-1]):
                key = tuple(parts[: idx + 1])
                if key not in folder_nodes:
                    folder_nodes[key] = self.tree.insert(parent_id, tk.END, text=part)
                parent_id = folder_nodes[key]
            # 文件节点
            self.tree.insert(
                parent_id,
                tk.END,
                text=parts[-1],
                values=(str(rel), d["doc"]),
                tags=("demofile",),
                iid=str(d["full"]),
            )
        self.status_var.set(f"共找到 {len(self.demo_list)} 个 *_demo.py")

    def get_selected_demo_path(self) -> Path | None:
        sel = self.tree.selection()
        if not sel:
            return None
        full_path = sel[0]
        try:
            file = Path(full_path)
            if file.exists() and file.is_file():
                return file
        except Exception:
            pass
        return None

    def run_selected(self):
        p = self.get_selected_demo_path()
        if p is None:
            self.status_var.set("请选中一个demo脚本")
            return
        self._spawn_process(p)

    def on_double_click(self, event):
        path = self.get_selected_demo_path()
        if path:
            self._spawn_process(path)

    def _spawn_process(self, script_path: Path):
        """新开独立进程运行demo，不阻塞GUI主窗口"""
        self.status_var.set(f"启动: {script_path.relative_to(BASE_DIR)}")
        # TODO：这里要优化下
        import sys

        try:
            subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(BASE_DIR.parent),
                # stdout=subprocess.DEVNULL,
                # stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            # TODO:
            # 后台线程读stdout、stderr，不卡tk主线程
            # import threading

            # threading.Thread(
            #     target=_read_pipe, args=(proc.stdout, self.on_stdout_line), daemon=True
            # ).start()
            # threading.Thread(
            #     target=_read_pipe, args=(proc.stderr, self.on_stderr_line), daemon=True
            # ).start()
        except Exception as e:
            self.status_var.set(f"启动失败: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DemoLauncher(root)
    root.mainloop()
