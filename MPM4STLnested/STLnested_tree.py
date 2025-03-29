import copy
import time

#  给两个列表取并集，考虑R以及\运算
#  给两个列表取并集，考虑R以及\运算
def bing(region1, region2):
    # 用来给两个集合取并集，考虑全集R，R的存法是region = ['R'] 存了一个字符R
    if region1[0] == 'R':
        region = region1
    elif region2[0] == 'R':
        region = region2
    else:  # 这种情况是两个都不是R，正常取并集即可
        if region1[0] == 'not':
            if region1[1] == 'R': # 如果1是['not', 'R']
                region = region2
            else: # 1不是['not', 'R']，后边是多边体
                if region2[0] == 'not':  # 都是not
                    if region2[1] == 'R': # 2是['not', 'R']
                        region = region1
                    else: # 后边都是多边体
                        region = ['not', region1[1].intersection(region2[1])]
                else:  # 1是not，2不是
                    region = ['not', region1[1].difference(region2[0])]
        elif region2[0] == 'not':  # 这时候region1肯定不是not了
            if region2[1] == 'R':  # 2是['not', 'R']
                region = region1
            else: # 2不是['not', 'R']，后边是多边体
                region = ['not', region2[1].difference(region1[0])]
        else:  # 俩都不是not不是R，正常取并集
            region = [region1[0].union(region2[0])]
    return region

#  给两个集合取交集，考虑R以及\运算
def jiao(region1, region2):
    # 用来给两个集合取交集，考虑全集R，R的存法是region = ['R'] 存了一个字符R
    if region1[0] == 'R':
        region = region2
    elif region2[0] == 'R':
        region = region1
    else:  # 这种情况是两个都不是R，正常取交集即可
        if region1[0] == 'not':
            if region2[0] == 'not':
                if region1[1] == 'R' or region2[1] == 'R': # 1或2是['not', 'R']的情况
                    region = ['not', 'R']
                else:
                    region = ['not', region1[1].union(region2[1])]
            else:
                if region1[1] == 'R': # 1是['not', 'R']的情况
                    region = ['not', 'R']
                else:
                    region = [region2[0].difference(region1[1])]
        elif region2[0] == 'not':  # 这时候region1肯定不是not了
            if region2[1] == 'R':  # 2是['not', 'R']的情况
                region = ['not', 'R']
            else:
                region = [region1[0].difference(region2[1])]
        else:  # 俩都不是not不是R，正常取交集
            # print(region1)
            # print(region2)
            region = [region1[0].intersection(region2[0])]
    return region

class SatTree():
    """一个能存某个STL_formula满足情况树结构的类"""
    def __init__(self, k, father=None, sat=None):
        self.k = k  # 当前时刻Ik的k
        self.sat = sat # 上一时刻的叶子节点满足情况(k-1时刻)
        self.father = father  # 前一个Ik-1
        self.child_list = []  # 后继集succ(Ik)
        self.tree_list = None
        # self.tree_list = self.get_tree_list()  # 用列表嵌套列表的方式表示树 [root_sat, [son, son,..., son]]
        # 不应该在构造的时候生成树，因为后边的点还没加进来，应该最后再弄这个
        self.feasible_set = None  # 用来存储当前时刻的可行集

    def __repr__(self):
        return self.draw()

    def set_tree_list(self):
        self.tree_list = self.get_tree_list()
        return self.tree_list

    def get_tree_list(self):
        tree_list = [f"k={self.k}{self.sat}", []]
        if self.child_list == []: # 这已经是最后的叶子[]了
            tree_list = [f"k={self.k}{self.sat}", None]
            return tree_list
        else:
            for i in range(len(self.child_list)):
                # tree_list[1] += [self.child_list[i].tree_list]
                tree_list[1] += [self.child_list[i].set_tree_list()]
            return tree_list

    # 画出对应的树结构
    def draw(self, tree_plot='', n=0, tree_list_now=None):
        if tree_list_now == None:
            tree_list_now = self.tree_list
        space =  '   '
        branch = '│  '
        tee =    '├─ '
        last =   '└─ '
        if tree_list_now[1] == None:  # 这已经是最后的叶子[]了
            pass
        else:
            if n == 0:
                tree_plot += str(tree_list_now[0]) + '\n'
            for i in range(len(tree_list_now[1])):
                if i != len(tree_list_now[1]) - 1:  # 还没到最后一个
                    tree_plot += n*space + tee + str(tree_list_now[1][i][0]) + '\n'
                    tree_plot += self.draw('', n+1, tree_list_now[1][i])
                else:
                    tree_plot += n*space + last + str(tree_list_now[1][i][0]) + '\n'
                    tree_plot += self.draw('', n+1, tree_list_now[1][i])
        return tree_plot


class Tree():
    """一个能存某个STL_formula对应的树结构的类"""
    def __init__(self, root, child_list=None, a_b=None):
        self.root = root  # 只能是 G/U/∧，如果是phi那就是[...]
        self.child_list = child_list  # 如果是G，那就只有一个Tree，如果是U就有俩，如果是∧那就想几个是几个，如果是phi就是None
        self.a_b = a_b  # 如果是G/U，才有，否则是None
        self.leaf_num = None  # 如果是phi即叶子节点才有，从0开始编号，否则是None
        self.leaf_under_list = None  # 下边都有哪些叶子节点
        self.tree_list = self.get_tree_list()  # 用列表嵌套列表的方式表示树 [root, [son, son,..., son]]

    def __repr__(self):
        return self.draw()

    def get_tree_list(self):
        tree_list = [self.root, []]
        if self.child_list == None:
            tree_list = [self.root, None]
            return tree_list
        else:
            for i in range(len(self.child_list)):
                tree_list[1] += [self.child_list[i].tree_list]
            return tree_list

    # 画出对应的树结构
    def draw(self, tree_plot='', n=0, tree_list_now=None):
        if tree_list_now == None:
            tree_list_now = self.tree_list
        space =  '   '
        branch = '│  '
        tee =    '├─ '
        last =   '└─ '
        if tree_list_now[1] == None:  # 这已经是最后的叶子[]了
            pass
        else:
            if n == 0:
                tree_plot += str(tree_list_now[0]) + '\n'
            for i in range(len(tree_list_now[1])):
                if i != len(tree_list_now[1]) - 1:  # 还没到最后一个
                    tree_plot += n*space + tee + str(tree_list_now[1][i][0]) + '\n'
                    tree_plot += self.draw('', n+1, tree_list_now[1][i])
                else:
                    tree_plot += n*space + last + str(tree_list_now[1][i][0]) + '\n'
                    tree_plot += self.draw('', n+1, tree_list_now[1][i])
        return tree_plot

    # 给这课树的叶子节点按print顺序从上到下编号（从0开始）
    def leaf_num_set(self, num=0):
        if self.child_list == None:
            self.leaf_num = num
            num += 1
        else:
            for i in range(len(self.child_list)):
                num = self.child_list[i].leaf_num_set(num)
        return num

    # 记录一下每个最外层运算符下边都有哪几个叶子节点
    def record_under_leaf(self, leaf_list=None):
        if leaf_list == None:  # 说明我现在就是最外边那一个，我需要记录self.leaf_under_list
            if self.root == '∧':  # 如果我是多个并列，那不用记录我的（反正是全部），转而记录我下边的每一个运算符的（真正的”最外层运算符“）
                for i in range(len(self.child_list)):
                    self.child_list[i].record_under_leaf()
            else:  # 如果我自己是最外层运算符
                self.leaf_under_list = []
                for i in range(len(self.child_list)):
                    self.leaf_under_list += self.child_list[i].record_under_leaf([])
        else:  # 我是内部的，我不需要记录，而是需要把我下边的叶子节点记录传回去   现在改了，现在内部的自己也记一下，有用
            if self.leaf_num != None:  # 说明我就是个叶子
                leaf_list += [self.leaf_num]
                return leaf_list
            else:  # 我不是叶子
                for i in range(len(self.child_list)):
                    leaf_list += self.child_list[i].record_under_leaf([])
                self.leaf_under_list = leaf_list
                return leaf_list

    # leaf_effective_horizon的辅助函数
    def add_ab(self, eh_list):
        eh_list_new = eh_list.copy()
        for i in range(len(eh_list_new)):
            eh_list_new[i][0] = eh_list_new[i][0] + self.a_b[0]
            eh_list_new[i][1] = eh_list_new[i][1] + self.a_b[1]
        return eh_list_new

    # 每个叶子的有效区间，用于后边生成所有情况
    def leaf_effective_horizon(self):
        leaf_list = []
        leaf_eh_list = []
        if self.child_list == None:  # 说明我就是个叶子
            # print(f"{self.root},{self.leaf_num}")
            leaf_list += [self.root]
            leaf_eh_list += [[0, 0]]
        else:  # 说明不是叶子，那就是G/U/∧
            if self.root == '∧':
                for i in range(len(self.child_list)):
                    leaf_list += self.child_list[i].leaf_effective_horizon()[0]
                    leaf_eh_list += self.child_list[i].leaf_effective_horizon()[1]
            else:
                for i in range(len(self.child_list)):
                    leaf_list += self.child_list[i].leaf_effective_horizon()[0]
                    leaf_eh_list += self.add_ab(self.child_list[i].leaf_effective_horizon()[1])
        return leaf_list, leaf_eh_list

    def check_U(self, qian_list, hou_list, a, b):
        if len(qian_list) > b-a+1:
            qian_list = qian_list[:b-a+1]
        if len(hou_list) > b-a+1:
            hou_list = hou_list[:b-a+1]
        num = min(len(qian_list), len(hou_list))
        for i in range(num):  # 如果hou直到最后一个都是False，就False了
            if hou_list[i] != False:
                break
            elif i == b-a:
                return False
        for i in range(num):
            if qian_list[i] == True:
                if hou_list[i] == True:
                    return True
                elif hou_list[i] == False:
                    if i == b-a:  # 如果已经是最后一个了，那就没了
                        return False
                    else:
                        continue
                elif hou_list[i] == None:
                    return None
            elif qian_list[i] == None:
                return None
            elif qian_list[i] == False:
                return False

    def check_G(self, g_list, a, b):
        if len(g_list) > b-a+1:
            g_list = g_list[:b-a+1]
        if len(g_list) < b-a+1:
            g_list += [None]
        for i in range(len(g_list)):
            if g_list[i] == False:  # 有False就False
                return False
            elif g_list[i] == None:  # 有None就None
                return None
        # 全是True就是True
        return True

    def check_conjuction(self, c_list):
        for i in range(len(c_list)):
            if c_list[i] == False:
                return False
        for i in range(len(c_list)):  # 到这里说明没有False，则不是True就是None
            if c_list[i] == None:
                return None
        return True

    #  让所有的叶子结点向后错位一个，其实就是直接把每个的第一个去掉，因为后边不需要管，在check_?里会自己截好的
    def p_cuowei1(self, p):
        p_cuowei = copy.deepcopy(p)
        for i in range(len(p_cuowei)):
            if len(p_cuowei[i]) == 1:
                p_cuowei[i] = [None]
            else:
                p_cuowei[i] = p_cuowei[i][1:]
        return p_cuowei

    def check(self, p, a_b_this=None):
        p = copy.deepcopy(p)
        if a_b_this == None:  # 说明我这个树是最外层的，不嵌套在某个运算符下
            if self.child_list == None:  # 这种情况基本不可能发生，因为我们一般不把单独的[]即phi放在最外层，至少要有一层时序逻辑运算符
                pass
            elif self.root == '∧':
                c_list = []
                for i in range(len(self.child_list)):
                    c_list += [self.child_list[i].check(p)]
                # print(c_list)
                return self.check_conjuction(c_list)
            elif self.root == 'G':
                if self.child_list[0].child_list == None:  # 如果是G且内层是phi而非Phi
                    return self.check_G(p[self.child_list[0].leaf_num], self.a_b[0], self.a_b[1])
                else:  # 内层是Phi
                    g_list = self.child_list[0].check(p, self.a_b)
                    return self.check_G(g_list, self.a_b[0], self.a_b[1])
            elif self.root == 'U':
                if self.child_list[0].child_list == None:  # 如果是U且内层(前)是phi而非Phi
                    qian_list = p[self.child_list[0].leaf_num]
                else:  # 内层(前)是Phi
                    qian_list = self.child_list[0].check(p, self.a_b)
                if self.child_list[1].child_list == None:  # 如果是U且内层(后)是phi而非Phi
                    hou_list = p[self.child_list[1].leaf_num]
                else:  # 内层(后)是Phi
                    hou_list = self.child_list[1].check(p, self.a_b)
                # print(f"这里是U的判断{qian_list},{hou_list},结果是{self.check_U(qian_list, hou_list, self.a_b[0], self.a_b[1])}")
                return self.check_U(qian_list, hou_list, self.a_b[0], self.a_b[1])
        else:  # 我是别人的内层，那应该传进来了一个a_b_this，得return一个长度为b-a+1的列表my_p回去
            my_p = []
            if self.child_list == None:  # 这种情况不可能发生，因为我们就不会对[]即phi的tree使用check函数，所以pass
                pass
            elif self.root == '∧':
                for st in range(a_b_this[1]-a_b_this[0]+1):
                    if st != 0:  # 除了第一个，后边的每次要错位一个
                        p = self.p_cuowei1(p)
                    c_list = []
                    for i in range(len(self.child_list)):
                        c_list += [self.child_list[i].check(p)]
                    my_p += [self.check_conjuction(c_list)]
                return my_p
            elif self.root == 'G':
                if self.child_list[0].child_list == None:  # 如果是G且内层是phi而非Phi
                    for st in range(a_b_this[1] - a_b_this[0] + 1):
                        if st != 0:  # 除了第一个，后边的每次要错位一个
                            p = self.p_cuowei1(p)
                        my_p += [self.check_G(p[self.child_list[0].leaf_num], self.a_b[0], self.a_b[1])]
                else:  # 内层是Phi
                    for st in range(a_b_this[1] - a_b_this[0] + 1):
                        if st != 0:  # 除了第一个，后边的每次要错位一个
                            p = self.p_cuowei1(p)
                        g_list = self.child_list[0].check(p, self.a_b)
                        my_p += [self.check_G(g_list, self.a_b[0], self.a_b[1])]
                return my_p
            elif self.root == 'U':  # !!!!!!!!这里现在还不是最快最优的形式，因为懒得写那么长了，先就这样，之后有空再改，改的跟G一样就行
                #  即if的判断放在for循环之前，这样就不必每次循环都再判断一次，但是代码会有点长!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                for st in range(a_b_this[1] - a_b_this[0] + 1):
                    if st != 0:  # 除了第一个，后边的每次要错位一个
                        p = self.p_cuowei1(p)
                    if self.child_list[0].child_list == None:  # 如果是U且内层(前)是phi而非Phi
                        qian_list = p[self.child_list[0].leaf_num]
                    else:  # 内层(前)是Phi
                        qian_list = self.child_list[0].check(p, self.a_b)
                    if self.child_list[1].child_list == None:  # 如果是U且内层(后)是phi而非Phi
                        hou_list = p[self.child_list[1].leaf_num]
                    else:  # 内层(后)是Phi
                        hou_list = self.child_list[1].check(p, self.a_b)
                    my_p += [self.check_U(qian_list, hou_list, self.a_b[0], self.a_b[1])]
                return my_p

    #  用来根据U的规则排除一些叶子节点的后续更新，把不需要继续更新的叶子编号传回去
    def paichu_U(self, p):
        no_need_update_leaf_list = []
        # 先提取Uqian和Uhou
        if self.child_list[0].child_list == None:  # 如果是U且内层(前)是phi而非Phi
            qian_list = p[self.child_list[0].leaf_num]
        else:  # 内层(前)是Phi
            qian_list = self.child_list[0].check(p, self.a_b)
        if self.child_list[1].child_list == None:  # 如果是U且内层(后)是phi而非Phi
            hou_list = p[self.child_list[1].leaf_num]
        else:  # 内层(后)是Phi
            hou_list = self.child_list[1].check(p, self.a_b)
        # 再根据Uqian和Uhou判断是否有不需要更新了的叶子节点
        for i in range(len(qian_list)):  # Uqian如果有False就不用更新了
            if qian_list[i] == False:
                if self.child_list[0].child_list == None:
                    no_need_update_leaf_list += [self.child_list[0].leaf_num]
                else:
                    no_need_update_leaf_list += self.child_list[0].leaf_under_list
        for i in range(len(hou_list)):  # Uhou如果有True就不用更新了
            if hou_list[i] == True:
                if self.child_list[1].child_list == None:
                    no_need_update_leaf_list += [self.child_list[1].leaf_num]
                else:  # 内层(后)是Phi
                    no_need_update_leaf_list += self.child_list[1].leaf_under_list
        return no_need_update_leaf_list


class Phi():
    """ 一个可嵌套的STL公式 """
    # 构造方法：
    def __init__(self, a, b, operator, phi, phi_1=[None]):  # 注意对U是先传后边再传前边，输入要按顺序来，G的phi_1虽没有也要给个None
        self.a = a  # 开始时刻
        self.b = b  # 结束时刻
        self.operator = operator  # 'U' or 'G'  这个U是按学姐论文的U'定义的
        self.phi = phi  # 后边那个
        self.phi_1 = phi_1  # 前边那个，仅U有
        self.Phi_list = []  # 当该Phi是好几个Phi的合的时候，用一个这个存储每一个单独的Phi

        if len(self.a) > 1:  # 如果是好几个并列的Phi组成的Phi，分别存储一下每个Phi(感觉有用)  (确实有用)
            for i in range(len(self.a)):
                self.Phi_list += [Phi([self.a[i]], [self.b[i]], [self.operator[i]], [self.phi[i]], [self.phi_1[i]])]

        self.tree = self.get_tree()
        self.tree.leaf_num_set()
        self.tree.record_under_leaf()

    # 重写print
    def __repr__(self):
        prt = ''
        for i in range(0,len(self.a)):
            if self.operator[i] == 'U':
                prt += f"({self.phi_1[i]}){self.operator[i]}[{self.a[i]},{self.b[i]}]({self.phi[i]})"
            elif self.operator[i] == 'G':
                prt += f"{self.operator[i]}[{self.a[i]},{self.b[i]}]({self.phi[i]})"
            if i != len(self.a)-1:
                prt += '∧'
        return prt

    # 生成Phi对应的树结构
    def get_tree(self):
        if len(self.operator) == 1:  # 如果只有一个，就不用∧了
            a_b = [self.a[0], self.b[0]]
            if self.operator[0] == 'G':
                if type(self.phi[0]) == type([]):
                    tree = Tree('G', [Tree(self.phi[0])], a_b)
                else:
                    tree = Tree('G', [self.phi[0].tree], a_b)
            if self.operator[0] == 'U':
                if type(self.phi_1[0]) == type([]) and type(self.phi[0]) == type([]):
                    tree = Tree('U', [Tree(self.phi_1[0]), Tree(self.phi[0])], a_b)
                elif type(self.phi_1[0]) == type(self) and type(self.phi[0]) == type(self):
                    tree = Tree('U', [self.phi_1[0].tree, self.phi[0].tree], a_b)
                elif type(self.phi_1[0]) == type([]) and type(self.phi[0]) == type(self):
                    tree = Tree('U', [Tree(self.phi_1[0]), self.phi[0].tree], a_b)
                elif type(self.phi_1[0]) == type(self) and type(self.phi[0]) == type([]):
                    tree = Tree('U', [self.phi_1[0].tree, Tree(self.phi[0])], a_b)
        else:  # 如果有多个，那就需要一个∧结点
            tree_list = []
            for i in range(len(self.Phi_list)):
                tree_list += [self.Phi_list[i].tree]
            tree = Tree('∧', tree_list)

        return tree

    # 计算一个公式的(最大)有效区间(对于G来说就是有效区间，对于U来说是最大有效区间)  输出[start,end]
    def effective_horizon(self):
        start_list = []
        end_list = []
        for i in range(0, len(self.a)):
            if self.operator[i] == 'G':
                if type(self.phi[i]) == type([]):  # 其实如果phi是一个[]，相当于是一个作用范围为[0,0]的Phi，可以看成是两个+0
                    start_list += [self.a[i]]
                    end_list += [self.b[i]]
                elif type(self.phi[i]) == type(self):
                    start_list += [self.a[i]+self.phi[i].effective_horizon()[0]]
                    end_list += [self.b[i]+self.phi[i].effective_horizon()[1]]

            elif self.operator[i] == 'U':  # a要加两边最小的a，一旦两边有一个是[]，相当于是0，所以+0，但b要加最大的b
                if type(self.phi_1[i]) == type([]) and type(self.phi[i]) == type([]):
                    start_list += [self.a[i]]
                    end_list += [self.b[i]]
                elif type(self.phi_1[i]) == type(self) and type(self.phi[i]) == type(self):
                    start_list += [self.a[i] + min([self.phi[i].effective_horizon()[0], self.phi_1[i].effective_horizon()[0]])]
                    end_list += [self.b[i] + max([self.phi[i].effective_horizon()[1], self.phi_1[i].effective_horizon()[1]])]
                elif type(self.phi_1[i]) == type([]) and type(self.phi[i]) == type(self):
                    start_list += [self.a[i]]
                    end_list += [self.b[i] + self.phi[i].effective_horizon()[1]]
                elif type(self.phi_1[i]) == type(self) and type(self.phi[i]) == type([]):
                    start_list += [self.a[i]]
                    end_list += [self.b[i] + self.phi_1[i].effective_horizon()[1]]
        # print(start_list)
        # print(end_list)
        start = min(start_list)
        end = max(end_list)
        return [start, end]

    def pailiezuhe(self, original_list_k, num, linshi_list=[], flag=0):
        result = []
        if flag < (num-1):
            for i in range(len(original_list_k[flag])):
                linshi_list_this = linshi_list + [original_list_k[flag][i]]
                result += self.pailiezuhe(original_list_k, num, linshi_list_this, flag+1)
        else:
            for i in range(len(original_list_k[flag])):
                linshi_list_this = linshi_list + [original_list_k[flag][i]]
                result += [linshi_list_this]
        return result

    # n个可能为T/F，生成2^n种排列方式
    def TF_pailiezuhe(self, n, linshi_list=[]):
        TF_list = copy.deepcopy(linshi_list)
        if TF_list == []:
            TF_list = [[True], [False]]
            TF_list = self.TF_pailiezuhe(n-1, TF_list)
        else:
            if n == 0:
                return TF_list
            else:
                anothor_list = copy.deepcopy(linshi_list)
                for i in range(len(TF_list)):
                    TF_list[i] += [True]
                    anothor_list[i] += [False]
                TF_list += anothor_list
                TF_list = self.TF_pailiezuhe(n-1, TF_list)
        return TF_list

    # 为一个p_list每一个需要加T/F的分别加T/F，如果有n个需要加，那就生成2^n个
    # p_list是 [[True, True,...], [True, False,...],..., [False, True,...]]
    def add_TF_p_list(self, p_list, index_list):
        n = len(index_list)
        if n == 0:
            return [p_list], None
        TF_list = self.TF_pailiezuhe(n)
        p_list_list = []
        for i in range(len(TF_list)):
            p_list_anothor = copy.deepcopy(p_list)
            for index_num in range(n):
                if p_list_anothor[index_list[index_num]] == [None]:
                    p_list_anothor[index_list[index_num]] = []
                p_list_anothor[index_list[index_num]] += [TF_list[i][index_num]]
            p_list_list += [p_list_anothor]
        return p_list_list, TF_list
    
    def add_None(self, n_leaf, TF_list, index_list):
        TF_list_add_None = []
        for i in range(len(TF_list)):
            TF_list_add_None += [[]]
            for j in range(n_leaf):
                if j in index_list:
                    TF_list_add_None[i] += [TF_list[i][index_list.index(j)]]
                else:
                    TF_list_add_None[i] += [None]
        return TF_list_add_None


    # 转变思路，不单独生成每个k时刻的all_p_k，从k=0开始往后生成，如果有已经死了的情况就直接中途毙掉，能活的再继续往后边加T/F，这样应该能好不少
    # !!!!!!!!!!!很关键，现在生成所有情况时，把空白的地方都用None填上!!!!!!!!!!!!!除非我能把上边的代码改好!!!!!!!!!!!!!!!!
    # 不用填了，改好了
    def all_p(self):
        leaf_list, leaf_eh_list = self.tree.leaf_effective_horizon()
        n_leaf = len(leaf_list)
        effctive = self.effective_horizon()
        all_p = []
        all_p_not_T = []
        all_p_k_last = []
        for i in range(len(leaf_list)):  # 生成一个初始的all_p_k_last [[[None], [None],..., [None]]]
            all_p_k_last += [[None]]
        all_p_k_last = [all_p_k_last]
        sat_tree_root = SatTree(0)  # 生成sat_tree的根节点，k=0时刻
        all_node_k_last = [sat_tree_root]
        for k in range(effctive[1]+2):
            print(k)
            all_p_k = []
            all_node_k = []
            addTF_index_list_k = []
            for i in range(len(leaf_list)):  # 首先统计哪几个leaf该时刻需要更新 leaf_eh_list[i][0]+1 =< k <= leaf_eh_list[i][1]+1
                if k <= leaf_eh_list[i][1]+1 and k >= leaf_eh_list[i][0]+1:
                    addTF_index_list_k += [i]
            print(addTF_index_list_k)
            # print(all_p_k_last)
            for i in range(len(all_p_k_last)):  # 把上一次所有的可能情况后边再添上这一刻可能的T/F，放到这一时刻的所有情况里
                addTF_index_list_k_this = addTF_index_list_k
                if self.tree.root == '∧':
                    for j in range(len(self.tree.child_list)):
                        if self.tree.child_list[j].check(all_p_k_last[i]) == True:
                            addTF_index_list_k_this = list(set(addTF_index_list_k_this)-set(self.tree.child_list[j].leaf_under_list))
                        elif self.tree.child_list[j].root == 'U':
                            addTF_index_list_k_this = list(set(addTF_index_list_k_this)-set(self.tree.child_list[j].paichu_U(all_p_k_last[i])))
                elif self.tree.root == 'U':
                    addTF_index_list_k_this = list(set(addTF_index_list_k_this)-set(self.tree.paichu_U(all_p_k_last[i])))
                all_p_k_add, sat_list_no_None = self.add_TF_p_list(all_p_k_last[i], addTF_index_list_k_this)
                all_p_k += all_p_k_add
                if sat_list_no_None == None:
                    all_node_k = all_node_k_last
                else:
                    sat_list = self.add_None(n_leaf, sat_list_no_None, addTF_index_list_k_this)
                    for j in range(len(sat_list)):
                        if self.tree.check(all_p_k_add[j]) == True:
                            # 已经完成的，直接连到上一层，不用加到all_node_k里，因为不用再往下走了
                            sat_tree_node = SatTree(k, all_node_k_last[i], sat_list[j])
                            all_node_k_last[i].child_list += [sat_tree_node]
                        elif self.tree.check(all_p_k_add[j]) != False:
                            # 还没完成的，连到上一层，并且加到all_node_k里，继续往下走
                            sat_tree_node = SatTree(k, all_node_k_last[i], sat_list[j])
                            all_node_k_last[i].child_list += [sat_tree_node]
                            all_node_k += [sat_tree_node]
            all_p_k_last = copy.deepcopy(all_p_k)
            all_node_k_last = all_node_k
            num = len(all_p_k)
            for i in range(num-1, -1, -1):  # 把死的给扔掉，只留下已经完成的和尚未完成的（已经完成的也不用再继续了吧？这怎么改？）!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # 必须要倒叙遍历，因为倒叙才能在即时pop的时候不影响后续的顺序
                # all_p_k里留下True的和None的
                # all_p_k_last里只留下None的
                check_result = self.tree.check(all_p_k[i])
                # print(all_p_k[i])
                # print(check_result)
                if check_result == False:
                    all_p_k.pop(i)
                    all_p_k_last.pop(i)
                elif check_result == True:
                    all_p_k_last.pop(i)
            all_p += [all_p_k]
            all_p_not_T += [all_p_k_last]
            print(f"全部情况数:{len(all_p_k)} 没完成继续的:{len(all_p_k_last)}")
        sat_tree_root.set_tree_list()
        return sat_tree_root

    def is_successor(self, p, p_next_k):
        for i in range(len(p)):
            if p[i] == [None]:  # 如果前一个是None，那后一个随便是啥必然是后继
                continue
            elif len(p[i]) == len(p_next_k[i]):  # 如果长度相同，那说明没加T/F，那得一样
                if p[i] != p_next_k[i]:
                    return False
            else:  # 如果长度不同，那前边的得一样
                if p[i] != p_next_k[i][:-1]:
                    return False
        return True

    def consistent_region(self, sat_tree_node):
        leaf_list, leaf_eh_list = self.tree.leaf_effective_horizon()
        sat = sat_tree_node.sat
        all_region = []
        for i in range(len(sat)):
            if sat[i] == None:  # 一样则没贡献
                continue
            else:  # 不一样就有贡献了
                if sat[i] == True:
                    all_region += [leaf_list[i]]
                elif sat[i] == False:
                    all_region += [['not', leaf_list[i][0]]]

        consistent_region = ['R']
        for region in all_region:
            consistent_region = jiao(consistent_region, region)

        return consistent_region


if __name__ == "__main__":
    o1 = Phi([1,2],[3,5],['G','U'],[[1,2],[1,3]],[None,[2,3]])
    print(o1)
    print(o1.tree)
    print(o1.tree.check([[True, True, True], [True, True, True, True], [False, False, True, False]]))
    print()
    # print(o1.all_p())
    # print()

    #
    # o5 = Phi([1],[5],['U'],[[3,4]],[[1,2,3]])
    # o8 = Phi([0],[2],['G'],[o5])
    # print(o8)
    # print(o8.effective_horizon())
    # print(o1)
    # print(o1.effective_horizon())
    #
    o3 = Phi([1],[2],['G'],[[5,10,15]])
    print(o3)
    print(o3.tree)
    print(o3.tree.check([[True, False, True]]))
    print()

    o7 = Phi([0],[1],['U'],[[10,100]],[o3])
    print(o7)
    print(o7.tree)
    t1 = time.time()
    o7.all_p()
    print(f"花费时间:{time.time() - t1}")
    print()

    o9 = Phi([3],[8],['G'],[[1,2]])
    # # print(o9)
    # o10 = Phi([2],[5],['G'],[o9])
    # print(o10)
    o11 = Phi([3],[8],['U'],[[1,2]],[o9])
    # print(o11)
    o12 = Phi([3],[8],['U'],[[1,2]],[o11])
    print(o12)
    print(o12.tree)
    print(o12.effective_horizon())
    print(o12.tree.leaf_effective_horizon())
    print()
    o12 = Phi([3], [8], ['U'], [[1, 2, 3]], [o1])
    print(o12)
    print(o12.tree)
    # print(o12.effective_horizon())
    print(o12.tree.leaf_effective_horizon())

    t1 = time.time()
    # print(o12.tree.check([[True, True, True, True, False, True, True, True],
    #                       [True, True, True, True, True, True, True, True, True],
    #                       [True, True, True, True, True, True, True, True, True],
    #                       [False, True, True]]))
    # o12.all_p()
    print(f"花费时间:{time.time() - t1}")

    o1 = Phi([0], [7], ['G'], [[1,2]])
    o2 = Phi([0], [2], ['G'], [[1,2]])
    # print(o1)
    # task = Phi([0,3],[2,5], ['U','G'], [o2,[1,2,3]], [[1,2,3,4],None])  # 在5分钟里到达[20,25]且到达后再维持7分钟
    # task = Phi([0], [4], ['U'], [o2], [[1, 2, 3, 4]])  # 在5分钟里到达[20,25]且到达后再维持7分钟
    task = Phi([0], [4], ['U'], [[1, 2, 3, 4]], [o2])  # 在5分钟里到达[20,25]且到达后再维持7分钟
    print(task)
    print(task.tree)
    # print(task.tree.leaf_under_list)
    # print(task.tree.child_list[0].leaf_under_list)
    # print(task.tree.child_list[1].leaf_under_list)
    print(task.effective_horizon())
    I_every_k = task.all_all_p()
    print(I_every_k)
    for i in range(task.effective_horizon()[1]+2):
        print("---------------------------------------------------------------")
        print(f"k={i}")
        print(f"{len(I_every_k[i])}")
        print(I_every_k[i])

