from typing import Sized


class HeterColumnarTable(Sized):
    def __len__(self):
        return len(self.data)

    def __init__(self, limit = 32):
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
            一个列表，包含指定列的所有元素。
        """
        return [row[column_index] for row in self.data]

    def get_all_columns(self):
        """获取所有列的数据。

        Returns:
            一个列表，每个元素是一个列表，表示一列的数据。
        """
        return tuple(zip(*self.data))


if __name__ == '__main__':
    table = HeterColumnarTable()
    table.add_row((1, 2.5, "apple"))
    table.add_row([4, 3.14, "banana"])

    # 获取第二列
    column2 = table.get_column(1)
    print(column2)  # 输出: [2.5, 3.14]

    # 获取所有列
    all_columns = table.get_all_columns()
    print(all_columns)  # 输出: [(1, 4), (2.5, 3.14), ('apple', 'banana')]
