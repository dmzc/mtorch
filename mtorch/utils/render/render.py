from mtorch.interfaces import IOperator, ITensor
from graphviz import Digraph
from pathlib import Path
import os
import time


def render(x: ITensor | list[ITensor], path: Path = None):

    # 有向图，从左到右流向LR
    dot = Digraph("signal_flow", format="png")
    dot.attr(
        rankdir="LR", style="filled", fontname="SimHei", fontsize="12", warning="0"
    )  # LR=从左到右
    dot.attr(
        "node",
        style="filled",
        fontname="SimHei",
        fontsize="12",
        fillcolor="white",
        align="left",
        margin="0.15,0.1",
    )
    IVariables: set = set()
    operators: set = set()
    handled_funtions: set = set()

    def add_node(node: IOperator, is_IVariable=True):
        id = node.id
        name = node.name
        if is_IVariable:
            if id not in IVariables:
                IVariables.add(id)
                IVariable: IVariable = node
                fill_color = "white"
                if IVariable.is_input:
                    fill_color = "green"
                dot.node(
                    name=id,
                    label=name,
                    shape="circle",
                    width="1.4",
                    fixedsize="1.2",
                    fillcolor=fill_color,
                )
        else:
            if id not in operators:
                operators.add(id)
                dot.node(
                    name=id,
                    label=name,
                    shape="box",
                    style="filled,rounded",
                    fillcolor="#87CEEB",
                )

    def add_edge(left: str, right: str, is_grad=False, label: str = None):
        if label is None:
            label = ""
        if is_grad:
            dot.edge(left, right, style="dashed", label=label)
        else:
            dot.edge(left, right, label=label)

    def process(func: IOperator):
        if func is None:
            return
        if func not in handled_funtions:
            handled_funtions.add(func)
        else:
            return

        inputs = func.inputs
        outputs = func.outputs

        func_id = func.id
        add_node(func, False)

        for output in outputs:
            add_node(output())
            add_edge(func_id, output().id)
            # add_edge(output_name, func_name, is_grad=True, label=f"{output.grad}")

        for input in inputs:
            add_node(input)
            add_edge(input.id, func_id)
            process(input.creator)

    if isinstance(x, list):
        for _x in x:
            process(_x.creator)
    else:
        process(x.creator)
    output_path = dot.render(view=True, quiet=True, cleanup=True)
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(dot.source)
    time.sleep(1)  # view为true时会调用外部工具显示图片，所以不能马上删除
    os.remove(output_path)
