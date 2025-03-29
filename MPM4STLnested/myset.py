import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class NDSet:
    def __init__(self, dimensions, precision, R_dimensions_list=[]):
        self.dimensions = dimensions
        self.precision = precision
        self.R_dimensions_list = R_dimensions_list
        self.set = set()

    # 把精度过高的点四舍五入到指定精度
    # def _round_point(self, point):
    #     return tuple(round(coord, self.precision) for coord in point)
    def _round_point(self, point):
        for coord in point:
            if coord == 'R':
                continue
            else:
                coord = round(coord, self.precision)
        return point

    # 添加一个点到集合中
    def add(self, point):
        if len(point) != self.dimensions:
            raise ValueError(f"Point must have {self.dimensions} dimensions")
        rounded_point = self._round_point(point)
        self.set.add(rounded_point)

    # 从集合中移除一个点
    def remove(self, point):
        rounded_point = self._round_point(point)
        self.set.discard(rounded_point)

    # # 判断一个点是否在集合中
    # def contains(self, point):
    #     if len(point) != self.dimensions:
    #         raise ValueError(f"Point must have {self.dimensions} dimensions")
    #     rounded_point = self._round_point(point)
    #     return rounded_point in self.set

    # 判断一个点是否在集合中，考虑某些维度为任意值'R'的情况
    def contains(self, point):
        if len(point) != self.dimensions:
            return False
        for p in self.set:
            if all(p[i] == 'R' or p[i] == point[i] for i in range(self.dimensions)):
                return True
        return False

    # 返回集合中所有点的列表
    def __repr__(self):
        return f"NDSet(dimensions={self.dimensions}, precision={self.precision}, set={self.set})"

    # 添加一个超矩形中的所有点到集合中，超矩形用bounds表示
    # bounds是一个列表，每个元素是一个元组，表示一个维度的范围，一个具体的例子如下：
    # bounds = [(1.0, 1.2), (2.0, 2.2), (3.0, 3.2)]
    def add_points_in_hyperrectangle(self, bounds):
        if len(bounds) != self.dimensions:
            raise ValueError(f"Bounds must have {self.dimensions} dimensions")

        def generate_points(bounds, current_point=[]):
            if len(current_point) == self.dimensions:
                self.add(tuple(current_point))
                return
            bound = bounds[len(current_point)]
            if bound == 'R':
                generate_points(bounds, current_point + ['R'])
            else:
                min_val, max_val = bound
                for val in np.arange(min_val, max_val + 10**-self.precision, 10**-self.precision):
                    generate_points(bounds, current_point + [round(val, self.precision)])
        
        generate_points(bounds)

    # 求两个集合的并集
    def union(self, other):
        if self.dimensions != other.dimensions or self.precision != other.precision:
            raise ValueError("Both sets must have the same dimensions and precision")
        result = NDSet(self.dimensions, self.precision)
        result.set = self.set.union(other.set)
        return result

    # 求两个集合的交集
    def intersection(self, other):
        if self.dimensions != other.dimensions or self.precision != other.precision:
            raise ValueError("Both sets must have the same dimensions and precision")
        result = NDSet(self.dimensions, self.precision)
        result.set = self.set.intersection(other.set)
        return result

    # 求两个集合的差集
    def difference(self, other):
        if self.dimensions != other.dimensions or self.precision != other.precision:
            raise ValueError("Both sets must have the same dimensions and precision")
        result = NDSet(self.dimensions, self.precision)
        result.set = self.set.difference(other.set)
        return result
    
    # 重载运算符 | ，实现两个集合的并集
    def __or__(self, other):
        if self.dimensions != other.dimensions or self.precision != other.precision:
            raise ValueError("Both sets must have the same dimensions and precision")
        result = NDSet(self.dimensions, self.precision)
        result.set = self.set.union(other.set)
        return result

    # 重载运算符 & ，实现两个集合的交集
    def __and__(self, other):
        if self.dimensions != other.dimensions or self.precision != other.precision:
            raise ValueError("Both sets must have the same dimensions and precision")
        result = NDSet(self.dimensions, self.precision)
        result.set = self.set.intersection(other.set)
        return result

    # 重载运算符 - ，实现两个集合的差集
    def __sub__(self, other):
        if self.dimensions != other.dimensions or self.precision != other.precision:
            raise ValueError("Both sets must have the same dimensions and precision")
        result = NDSet(self.dimensions, self.precision)
        result.set = self.set.difference(other.set)
        return result
    
    # 当集合是2D或3D时，可以调用这个函数绘制集合中的点
    def plot(self):
        if self.dimensions == 2:
            self._plot_2d()
        elif self.dimensions == 3:
            self._plot_3d()
        else:
            raise ValueError("Plotting is only supported for 2D and 3D sets")

    def _plot_2d(self):
        x, y = zip(*self.set)
        plt.scatter(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D NDSet')
        plt.show()

    def _plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = zip(*self.set)
        ax.scatter(x, y, z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D NDSet')
        plt.show()

    # 矩阵和向量相乘
    def matrix_vector_multiply(self, A, x):
        if len(x) != len(A) or any(len(row) != len(x) for row in A):
            raise ValueError("Matrix A must be n*n and vector x must be n-dimensional")
        result = np.dot(A, x)
        return tuple(result)
    
    # 创建一个超矩形，x是超矩形的中心，bounds是超矩形的边界
    def create_hyperrectangle(self, x, bounds):
        if len(x) != self.dimensions or len(bounds) != self.dimensions:
            raise ValueError(f"Point and bounds must have {self.dimensions} dimensions")

        hyperrectangle = NDSet(self.dimensions, self.precision)

        def generate_points(bounds, current_point=[]):
            if len(current_point) == self.dimensions:
                hyperrectangle.add(tuple(current_point))
                return
            min_val, max_val = bounds[len(current_point)]
            base_val = x[len(current_point)]
            for val in np.arange(base_val + min_val, base_val + max_val + 10**-self.precision, 10**-self.precision):
                generate_points(bounds, current_point + [round(val, self.precision)])

        generate_points(bounds)
        return hyperrectangle
    
    # 把集合中的所有点通过矩阵A进行变换，再套上一个超矩形，然后求并集
    # A是一个n*n的矩阵，bounds是超矩形的边界，sparsify_precision是精度，用于控制计算量
    # 这个函数的实现逻辑是：对集合中的每个点，通过矩阵A变换，然后生成一个超矩形，最后求并集
    # 如果提供了sparsify_precision，会先对集合进行稀疏化，然后再计算
    # sparsify_precision越大，计算量越小，但是精度越低，默认是None，表示不稀疏化
    # 事实上，只要你的sparsify_precision和bounds的范围匹配，就不会影响结果
    # ---------------------------------------------------------------------
    # 注意，自动稀疏化并不总是对的，只有在A=I的时候才是对的，否则可能会导致结果不准确  x说错了
    # 所以在调用这个函数的时候，需要注意是否需要自动稀疏化，如果不需要，就手动提供稀疏化精度
    # 在A=I的情况下也不一定是对的，但是具体为什么现在还不清楚，总之先把默认值改成False了
    # 在之前的基础上加了个除以二，现在又对了，那就先用着，可能之前步子迈的太大了
    # ---------------------------------------------------------------------
    # 更新一下，现在是通过A的逆矩阵来变换集合中的点，并且不是简单的套上超矩形
    # 而是套上超矩形后再用A的逆矩阵变换一次，即现在已经直接实现了算OneStepSet
    # 但输入还是输入A和BU，求逆和套变换后的超矩形的逻辑在这个函数里实现
    # 好吧，我并不知道该怎么自洽的实现，所以这个函数暂时不实现，现在只能处理A=I的情况
    def transform_and_union_hyperrectangles(self, A, bounds, auto_sparsify=False, sparsify_precision=None):
        if len(A) != self.dimensions or any(len(row) != self.dimensions for row in A):
            raise ValueError("Matrix A must be n*n and match the dimensions of the set")
        if len(bounds) != self.dimensions:
            raise ValueError(f"Bounds must have {self.dimensions} dimensions")

        # A_inv = np.linalg.inv(A)
        A_inv = A  # 这里先不实现求逆矩阵的逻辑，先假设A是单位矩阵

        if auto_sparsify:
            # 如果自动稀疏化，用bounds的范围来计算稀疏化精度
            sparsify_precision = max(min(max_val - min_val for min_val, max_val in bounds)/2, 10**-self.precision)
            sparsified_set = self.sparsify(sparsify_precision)
        else:
            # 如果没有提供 sparsify_precision，用整个集合的点来计算
            if sparsify_precision is None:
                sparsified_set = self
            else:
                # 如果提供了 sparsify_precision，用提供的精度来稀疏化集合
                sparsified_set = self.sparsify(sparsify_precision)

        result_set = NDSet(self.dimensions, self.precision)
        for point in sparsified_set.set:
            transformed_point = self.matrix_vector_multiply(A_inv, point)
            hyperrectangle = self.create_hyperrectangle(transformed_point, bounds)
            result_set = result_set | hyperrectangle

        return result_set
    
    # 返回按照坐标排序的集合
    def sorted_points(self):
        return sorted(self.set)
    
    # 把集合中的点按照指定精度进行稀疏化
    def sparsify(self, sparsify_precision):
        sparsified_set = NDSet(self.dimensions, self.precision)
        for point in self.set:
            sparsified_point = tuple(round(coord / sparsify_precision) * sparsify_precision for coord in point)
            sparsified_set.add(sparsified_point)
        return sparsified_set

    # def is_on_edge(self, point):
    #     # 这里实现判断点是否在超多面体边缘上的逻辑
    #     # 这个函数需要根据具体的超多面体定义来实现
    #     pass

    # def get_edge_points(self):
    #     edge_points = set()
    #     for point in self.set:
    #         if self.is_on_edge(point):
    #             edge_points.add(point)
    #     return edge_points


if __name__ == "__main__":
    # Example usage:
    # nd_set1 = NDSet(3, 1)
    # bounds1 = [(1.0, 2), (2.0, 3), (3.0, 4)]
    # nd_set1.add_points_in_hyperrectangle(bounds1)
    # print(nd_set1)
    # nd_set2 = NDSet(3, 1)
    # bounds2 = [(1.5, 2.5), (2.5, 3.5), (3.5, 4.5)]
    # nd_set2.add_points_in_hyperrectangle(bounds2)
    # print(nd_set2)
    # print(nd_set1|nd_set2)
    # print(nd_set1&nd_set2)
    # print(nd_set1-nd_set2)
    # nd_set1.plot()
    # nd_set2.plot()
    # (nd_set1|nd_set2).plot()
    # (nd_set1&nd_set2).plot()
    # (nd_set1-nd_set2).plot()

    # Example usage:
    t0 = time.time()
    nd_set = NDSet(3, 1)
    bounds = [(1.0, 2), (2.0, 3), 'R']
    nd_set.add_points_in_hyperrectangle(bounds)
    print(nd_set)
    print(nd_set.contains((1.5, 2.5, 1000)))
    # x = (1.0, 2.0, 3.0)
    # bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
    # hyperrectangle = nd_set.create_hyperrectangle(x, bounds)
    # print(hyperrectangle)
    # hyperrectangle.plot()
    # A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
    # transformed_set = nd_set.transform_and_union_hyperrectangles(A, bounds)
    # # print(transformed_set)
    # print(time.time() - t0)
    # transformed_set.plot()

    # # Sparsify the set
    # sparsified_set = nd_set.sparsify(0.3)
    # print(sparsified_set)
    # sparsified_set.plot()
    pass
