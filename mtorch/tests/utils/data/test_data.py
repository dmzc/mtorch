from mtorch.utils.data import Dataset, DataLoader
import numpy as np
import pytest


class MockDataset(Dataset):

    mock_data: np.ndarray
    mock_label: np.ndarray

    def __init__(
        self, enable_label=False, data: np.ndarray = None, label: np.ndarray = None
    ):
        if data is not None and not isinstance(data, np.ndarray):
            data = np.array(data)
        if label is not None and not isinstance(label, np.ndarray):
            label = np.array(label)
        super().__init__(enable_label)
        self.mock_data = data
        self.mock_label = label

    def load_data(self) -> tuple[np.ndarray] | np.ndarray:

        if self.mock_label is not None:
            return (self.mock_data, self.mock_label)
        else:
            return self.mock_data


def test_dataset():

    dataset = MockDataset(enable_label=True, data=[[1, 2], [3, 4]])
    with pytest.raises(ValueError) as exec_info:
        dataset[1]
    assert "Enable_label is True,label must be returned." in str(
        exec_info.value
    ), "enable_label为True时必须返回标签数据"
    dataset = MockDataset(
        enable_label=True, data=[[1, 2], [3, 4], [5, 6]], label=[1, 2, 3]
    )
    assert len(dataset) == 3, "数据集长度正常获取"
    data: tuple[np.ndarray] = dataset[0:1]
    assert data[0].tolist() == [[1, 2]] and data[1].tolist() == [
        1,
    ], "enable_label为True时能正常返回标签数据"

    data = dataset[2:10]
    assert len(data[0]) == 1, "部分超出索引范围，只取没超出的"
    data = dataset[6:10]
    assert len(data[0]) == 0, "都超出索引范围，返回空"


def test_dataLoader():
    dataset = MockDataset(
        enable_label=True,
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
            data: tuple[np.ndarray] = data
            result.append((data[0].tolist(), data[1].tolist()))
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
