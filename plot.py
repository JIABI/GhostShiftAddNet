import matplotlib.pyplot as plt
import numpy as np
with open('temp/shift_ps_20_wb_5_add-32/record.txt') as f:
    data = f.read()
    data = data.split('\n')
    trainloss = [row.split(',')[0] for row in data]
    #trainacc1 = [(line.split()[1]) for line in lines]
    #testacc1 = [(line.split()[2]) for line in lines]
    #traintime = [(line.split()[3]) for line in lines]
    #testtime = [(line.split()[4]) for line in lines]
    #shifts = [(line.split()[5]) for line in lines]
    index = [i for i, val in enumerate(trainloss)]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("Plot DAta")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xticklabels(index)
    ax1.plot(trainloss, c='r', label='the data')
    leg = ax1.legend()
    plt.locator_params(nbins=len(index)-1)
    plt.show()
