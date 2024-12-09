import matplotlib.pyplot as plt
import os

'''
def loss_plot(args, num, loss):
    # num = args.epoch
    x = [i for i in range(num)]
    plot_save_path = r'./train_result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(
        args.epoch) + '_loss.jpg'
    plt.figure()
    plt.plot(x, loss, label='loss')
    plt.legend()
    plt.savefig(save_loss)


def metrics_plot(arg, name, *args):
    num = arg.epoch
    names = name.split('&')
    metrics_value = args
    i = 0
    x = [i for i in range(num)]
    plot_save_path = r'./train_result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(arg.arch) + '_' + str(arg.batch_size) + '_' + str(arg.dataset) + '_' + str(
        arg.epoch) + '_' + name + '.jpg'
    plt.figure()
    for l in metrics_value:
        plt.plot(x, l, label=str(names[i]))
        # plt.scatter(x,l,label=str(l))
        i += 1
    plt.legend()
    plt.savefig(save_metrics)
'''


def loss_plot(args, loss_list, plot_save_path, name='loss'):
    #num = args.epoch
    num = len(loss_list)
    x = [i for i in range(num)]
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path + str(args.dataset) + '_' + str(args.model) + '_' + str(args.epoch) + '_' + str(
        args.batch_size) + '_loss.jpg'
    plt.figure()
    plt.plot(x, loss_list, label=name)
    plt.legend()
    plt.savefig(save_loss)


def metrics_plot(args, metrics_list, plot_save_path, name):
    # num = args.epoch
    num = len(metrics_list)
    names = name.split('&')
    metrics_value = metrics_list
    i = 0
    x = [i for i in range(num)]

    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    save_metrics = plot_save_path + str(args.dataset) + '_' + str(args.model) + '_' + str(args.epoch) + '_' + str(
        args.batch_size) + '_' + name + '.jpg'

    plt.figure()
    plt.plot(x, metrics_list, label=name)
    plt.legend()
    plt.savefig(save_metrics)


# 可以将多个指标画在同一张图上
def metrics_plots(arg, name, plot_save_path, *args):
    # num = arg.epoch

    names = name.split('&')
    metrics_value = args
    i = 0

    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(arg.dataset) + '_' + str(arg.model) + '_' + str(arg.epoch) + '_' + str(
        arg.batch_size) + '_' + name + '.jpg'
    plt.figure()
    for l in metrics_value:
        num = len(l)
        x = [i for i in range(num)]
        plt.plot(x, l, label=str(names[i]))
        # plt.scatter(x,l,label=str(l))
        i += 1
    plt.legend()
    plt.savefig(save_metrics)


# 绘制一个简单的折线图，其中横轴表示列表元素个数，纵轴表示列表元素的值
def epoch_lr_bar():
    import matplotlib
    import matplotlib.pyplot as plt
    # 以示例一的数据作图
    a = ['a','b','c','d']

    b = [1,2,3,4]

    # font = {
    #     'family': 'SimHei',
    #     'weight': 'bold',
    #     'size': 12
    # }
    # matplotlib.rc("font", **font)
    # 设置图片大小
    plt.figure(figsize=(14, 8))
    # 绘制条形图
    plt.bar(a, b)
    # 设置刻度及刻度标签
    x_t = list(range(len(a)))
    plt.xticks(x_t, a, rotation=90)
    plt.show()


def legend_use():
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.random.uniform(-1, 1, 4)
    y = np.random.uniform(-1, 1, 4)
    p1, = plt.plot([1, 2, 3])
    p2, = plt.plot([3, 2, 1])

    l1 = plt.legend([p2, p1], ["line 2", "line 1"], loc='upper left')
    p3 = plt.scatter(x[0:2], y[0:2], marker='D', color='r')
    p4 = plt.scatter(x[2:], y[2:], marker='D', color='g')
    # This removes l1 from the axes.
    plt.legend([p3, p4], ['label', 'label1'], loc='lower right', scatterpoints=1)
    # Add l1 as a separate artist to the axes
    plt.gca().add_artist(l1)
    plt.show()


def legend_use2():
    import matplotlib.pyplot as plt
    line1, = plt.plot([1, 2, 3], label="Line 1", linestyle='--')
    line2, = plt.plot([3, 2, 1], label="Line 2", linewidth=4)
    # 为第一个线条创建图例
    first_legend = plt.legend(handles=[line1], loc=1)
    # 手动将图例添加到当前轴域
    ax = plt.gca().add_artist(first_legend)
    # 为第二个线条创建另一个图例
    plt.legend(handles=[line2], loc=4)
    plt.show()


# 绘制一个简单的折线图，其中横轴表示列表元素个数，纵轴表示列表元素的值
def plot():
    import matplotlib.pyplot as plt
    # 给定列表
    lst = [1, 10, 2, 9, 3, 8, 4]
    # 列表元素个数
    n = len(lst)
    # 横轴：列表元素个数,只要x轴和y轴维度相同即可，数据类型不同无关系
    # x = (1,2,3,4,5,6,7)
    # x=[1,2,3,4,5,6,7]
    # x = range(1, n + 1)
    # x = [i for i in range(1, n + 1)]

    # 纵轴：列表元素,也可以只在纵轴内放置一个列表或元组，如plt.plot([1,2,3,4])

    y = lst
    x = [1, 10, 20, 30, 40, 50, 60]
    # 绘制折线图
    plt.plot(x, y)
    # 添加标题和标签
    plt.title("List Elements")
    plt.xlabel("Number of Elements")
    plt.ylabel("List Element")
    # 显示图形
    plt.show()


def save():
    i = 0
    x = [i for i in range(100)]
    plot_save_path = r'./plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + "shiyan" + '.jpg'

    plt.savefig(save_metrics)
