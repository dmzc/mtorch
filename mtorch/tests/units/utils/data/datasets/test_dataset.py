from mtorch._interfaces import DatasetData
from mtorch.tests._utils import MockDataset


def test_dataset():

    dataset = MockDataset(data=[[1, 2], [3, 4], [5, 6]], label=[1, 2, 3])
    assert len(dataset) == 3, "数据集长度正常获取"
    data: DatasetData = dataset[0:1]
    assert data.data.tolist() == [[1, 2]] and data.label.tolist() == [
        1,
    ], "索引为范围切片值数据正常获取"

    data: DatasetData = dataset[0]
    assert data.data.tolist() == [[1, 2]] and data.label.tolist() == [
        1,
    ], "索引为数字数据正常获取"

    data: DatasetData = dataset[[0, 2]]
    assert data.data.tolist() == [[1, 2], [5, 6]] and data.label.tolist() == [
        1,
        3,
    ], "索引为数组正常获取"

    # data: DatasetData = dataset[0, 1]
    # assert data.data.tolist() == [[1, 2]] and data.label.tolist() == [
    #     1,
    # ], "索引为多维度切片且第一维为数值，只有第一维度生效"

    # data: DatasetData = dataset[1:3, 1]
    # assert data.data.tolist() == [[3, 4], [5, 6]] and data.label.tolist() == [
    #     2,
    #     3,
    # ], "索引为多维度切片且第一维为切片，只有第一维度生效"

    data = dataset[2:10]
    assert len(data.data) == 1, "部分超出索引范围，只取没超出的"
    data = dataset[6:10]
    assert len(data.data) == 0, "都超出索引范围，返回空"
