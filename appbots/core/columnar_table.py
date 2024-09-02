import numpy as np


class ColumnarTable:
    def __init__(self, limit: int = 32):
        self.data = []
        self.limit = limit

    def add_row(self, row):
        """添加一行数据。

        Args:
            row: 一个可迭代对象，表示一行数据。
        """
        if len(self.data) >= self.limit:
            self.data.pop(0)

        self.data.append(list(row))

    def get_column(self, column_index):
        """获取指定列的数据。

        Args:
            column_index: 列索引，从0开始。

        Returns:
            一个NumPy数组，表示指定列的数据。
        """
        return np.array([row[column_index] for row in self.data])

    def get_all_columns(self):
        """获取所有列的数据。

        Returns:
            一个NumPy数组，每一列对应一个原始列表中的列。
        """
        return np.array(self.data).T


if __name__ == "__main__":
    table = ColumnarTable()
    table.add_row((1, 2, 4))
    table.add_row((3, 5, 6))
    table.add_row((7, 8, 9))

    # 获取第二列
    column2 = table.get_column(1)
    print(column2)  # 输出：array([2, 5, 8])

    # 获取所有列
    a, b, c = table.get_all_columns()
    print(a, b, c)
