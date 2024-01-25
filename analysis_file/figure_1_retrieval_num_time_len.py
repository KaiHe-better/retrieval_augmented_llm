import matplotlib.pyplot as plt
import numpy as np

num =                   [0, 1,      2,      3,     4,     5,     6,     7,     8,      9,    10 ]

standard_RA_len   =     [0, 501,   1010,   1516,   2021,  2529, 3035,  3544,  4052,   4555,  5062]
AD_RA_len   =           [0, 501,   1010,   1516,   2021,  2529, 3035,  3544,  4052,   4555,  5062]


# standard_RA_7B_ACC  = [35.19, 36.45, 36.06, 36.76, 36.92, 36.53, 30.56, 12.65, 2.91, 1.26]

standard_RA_7B_ACC  =   [31.89, 34.64, 36.14, 36.45, 36.61,  36.76, 36.45, 30.87,  12.96, 3.06,  1.02]
standard_RA_7B_TIME  =  [1.5, 3.52,  5.73,  7.22,  10.43,  13.5,  17.92, 20.73,  24.87, 32.82, 44.77]
standard_RA_7B_hal   =  [0,   0.0,   0.0,   0.0,   0.0 ,   0.0 ,  00.15, 7.93,   54.35, 89.70, 96.38]

# standard_RA_13B =      [36.68, 37.08, 37.47, 39.12, 38.49, 37.78, 36.37, 23.57, 6.91, 1.41]

standard_RA_13B =        [36.29, 36.48, 37.00, 37.08, 38.05, 38.48, 37.55, 36.76, 23.88, 6.83, 1.57 ]
standard_RA_13B_TIME  =  [2.1,   4.5,   7.73,  11.22, 15.6,  20.3,  25.53, 31.37, 37.85, 51.4, 67.13]
standard_RA_13B_hal   =  [0,     0.0,   0.0,   0.0,   0.0 ,  0.0 ,  00.15, 00.78, 26.94, 81.93,95.67]

# standard_RA_70B =      [45.72, 46.82, 47.52, 48.72, 48.31, 48.69, XX,    XX,    XX,  XX ]

standard_RA_70B =        [44.78, 45.40, 46.11, 46.19, 48.23, 48.47, 48.69, 47.45, 40.53, 22.39, 7.15]
standard_RA_70B_TIME  =  [6.92, 16.17, 27.98, 40.0,  57.23, 73.35, 91.45, 111.5, 192.4, 176.43,229.77]
standard_RA_13B_hal   =  [0, 0.0,  0.0,    0.0,   0.0 ,   0.0 ,  0.0,  00.07, 1.09,  27.57, 78.71 ]





def col_1_1():
    # number of hops of EntailmentBank
    fig, ax1 = plt.subplots(1,1,figsize=(8, 3.6),dpi=300)
    plt.subplots_adjust(left=0.07, right=0.993, top=0.98, bottom=0.135)

    plt.grid(zorder=-100)

    bar_width = 0.35
    tick_label = num
    x = num

    save_file = "7b-1-1.jpg" 
    ax1_xlabel = "The number of retrieval documents"

    ax1_ylabel = "Performance (%)"
    bar1= standard_RA_7B_ACC
    bar2= [i+1.5 for i in bar1]
    # ax1.set_ylim(25,50)
    ax1.set_ylim(0,50)

    ax1.bar([i-bar_width/2 for i in x], bar1, bar_width, align="center", color='lightsteelblue', edgecolor='black', zorder=100, hatch='')
    ax1.bar([i+bar_width/2 for i in x], bar2, bar_width, align="center", color='coral', edgecolor='black', zorder=100, hatch='//')

    ax1.set_ylabel(ax1_ylabel,size=14)

    plt.yticks(size=11)


    plt.xticks(x,tick_label,size=12)
    plt.yticks(size=11)
    ax1.set_xlabel(ax1_xlabel,size=14)
    plt.legend(['  Standard-RA','  AD-RA'], fontsize=10)
    plt.savefig(save_file)


def col_2_1():
    
    # number of hops of EntailmentBank
    fig, ax1 = plt.subplots(1,1,figsize=(8, 3.6),dpi=300)
    plt.subplots_adjust(left=0.1, right=0.92, top=0.98, bottom=0.135)

    plt.grid(zorder=-100)

    bar_width = 0.35
    tick_label = num
    x = num

    save_file = "7b-2-1.jpg" 
    ax1_xlabel = "The number of retrieval documents"

    ax1_ylabel = "The number of retrieval tokens"
    bar1= standard_RA_len
    bar2= [i/2 for i in bar1]
    ax1.set_ylim(0, 5000)


    ax1.bar([i-bar_width/2 for i in x], bar1, bar_width, align="center", color='lightsteelblue', edgecolor='black', zorder=100, hatch='')
    ax1.bar([i+bar_width/2 for i in x], bar2, bar_width, align="center", color='coral', edgecolor='black', zorder=100, hatch='//')

    ax1.set_ylabel(ax1_ylabel,size=14)
    ax1.set_xlabel(ax1_xlabel,size=14)
    plt.yticks(size=11)
    plt.legend(['  Standard-RA','  AD-RA'], fontsize=10, loc='upper left')

    # ====================================================================================================================
    ax2 = ax1.twinx()
    ax2_ylabel = "Run time (mins.)"
    ax2.set_ylim(0, 250)

    line1 = standard_RA_7B_TIME
    line2 = [i-10 for i in line1]

    ax2.plot(x, line1, color="green",marker="o",linestyle='-.')
    ax2.plot(x, line2, color="red",marker="s",linestyle='--')
    ax2.set_ylabel(ax2_ylabel,size=14)
    

    plt.xticks(x,tick_label,size=12)
    plt.yticks(size=11)
    
    plt.legend(['  Standard-RA','  AD-RA'], fontsize=10)
    plt.savefig(save_file)


def col_3_1():
    # number of hops of EntailmentBank
    fig, ax1 = plt.subplots(1,1,figsize=(5, 3.6),dpi=300)
    plt.subplots_adjust(left=0.09, right=0.993, top=0.98, bottom=0.135)

    plt.grid(zorder=-100)

    bar_width = 0.35
    num = [6,     7,     8,      9,    10 ]
    tick_label = num
    x = num

    save_file = "7b-3-1.jpg" 
    ax1_xlabel = "The number of retrieval documents"

    ax1_ylabel = "The ratio of hallucination (%)"
    bar1= standard_RA_7B_hal[num[0]:]
    bar2= [i/2 for i in bar1]
    ax1.set_ylim(0,100)

    ax1.bar([i-bar_width/2 for i in x], bar1, bar_width, align="center", color='lightsteelblue', edgecolor='black', zorder=100, hatch='')
    ax1.bar([i+bar_width/2 for i in x], bar2, bar_width, align="center", color='coral', edgecolor='black', zorder=100, hatch='//')

    ax1.set_ylabel(ax1_ylabel,size=14)

    plt.yticks(size=11)


    plt.xticks(x,tick_label,size=12)
    plt.yticks(size=11)
    ax1.set_xlabel(ax1_xlabel,size=14)
    plt.legend(['  Standard-RA','  AD-RA'], fontsize=10)
    plt.savefig(save_file)


col_1_1()
# col_2_1()
# col_3_1()