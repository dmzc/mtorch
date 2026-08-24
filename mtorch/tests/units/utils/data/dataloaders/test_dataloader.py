from mtorch.tests._utils import MockDataset
from mtorch._interfaces import DatasetData
from mtorch.utils.data.dataloaders import DataLoader
import numpy as np


def test_dataLoader():
    dataset = MockDataset(
        data=[
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
            [19, 20],
            [21, 22],
        ],
        label=[1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 1],
    )
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=3)

    def get_result() -> list[tuple[np.ndarray]]:
        result: list[tuple[np.ndarray]] = []
        for data in dataloader:
            data: DatasetData = data
            result.append((data.data.tolist(), data.label.tolist()))
        return result

    assert get_result() == [
        ([[1, 2], [3, 4], [5, 6]], [1, 2, 3]),
        ([[7, 8], [9, 10], [11, 12]], [1, 2, 1]),
        ([[13, 14], [15, 16], [17, 18]], [2, 1, 2]),
        ([[19, 20], [21, 22]], [1, 1]),
    ], "默认不对截断数据做处理"

    dataloader = DataLoader(
        dataset=dataset, shuffle=False, batch_size=3, drop_last=True
    )
    assert get_result() == [
        ([[1, 2], [3, 4], [5, 6]], [1, 2, 3]),
        ([[7, 8], [9, 10], [11, 12]], [1, 2, 1]),
        ([[13, 14], [15, 16], [17, 18]], [2, 1, 2]),
    ], "drop_last为true丢弃截断数据"

    def complete_seq(indexes: list[int], complete_size: int, batch_size: int):
        """
        从头按顺序补全
        """
        return indexes[0:complete_size]

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=3,
        complete_strategy=complete_seq,
    )
    assert get_result() == [
        ([[1, 2], [3, 4], [5, 6]], [1, 2, 3]),
        ([[7, 8], [9, 10], [11, 12]], [1, 2, 1]),
        ([[13, 14], [15, 16], [17, 18]], [2, 1, 2]),
        ([[19, 20], [21, 22], [1, 2]], [1, 1, 1]),
    ], "传递compelete_strategy补全截断数据"
    dataloader = DataLoader(
        dataset=dataset, shuffle=False, batch_size=3, max_iteration=2
    )
    assert get_result() == [
        ([[1, 2], [3, 4], [5, 6]], [1, 2, 3]),
        ([[7, 8], [9, 10], [11, 12]], [1, 2, 1]),
    ], "限制迭代轮数"
    dataloader = DataLoader(
        dataset=dataset, shuffle=True, batch_size=3, max_iteration=1
    )
    result1 = get_result()
    result2 = get_result()
    assert result1 != result2, "shuffle为True时，每轮迭代都会打乱结果"

    dataloader = DataLoader(dataset=dataset, shuffle=False)
    assert get_result() == [
        (
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
                [13, 14],
                [15, 16],
                [17, 18],
                [19, 20],
                [21, 22],
            ],
            [1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 1],
        )
    ], "不设置batch_size则一次性返回数据"
