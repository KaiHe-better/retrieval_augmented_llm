import matplotlib.pyplot as plt
import numpy as np

num =                   [0,    1,      2,      3,     4,     5,    6,     7,     8,      9,    10 ]

standard_RA_len   =     [0,    501,   1010,   1516,  2021,  2529, 3035,  3544,  4052,   4555,  5062]
AD_RA_len   =           [0,    534,   978,    1930,  1890,  1519, 1484,  2437,  2479,   3126,  4066]



standard_RA_7B_ACC  =   [31.89, 34.64, 36.14, 36.45, 36.61,  36.76, 36.45, 30.87,  12.96, 3.06,  1.02]
standard_RA_7B_TIME  =  [1.50,  3.52,  5.73,  7.22,  10.43,  14.5,  17.92, 20.73,  24.87, 32.82, 44.77]
standard_RA_7B_hal   =  [0.0,   0.0,   0.0,   0.0,   0.0 ,   0.0 ,  00.15, 7.93,   54.35, 89.70, 96.38]

AD_RA_7B_ACC   =        [31.89, 36.61, 36.37, 36.61, 37.78,  39.02, 38.18, 36.53,  34.41, 30.87, 16.58]
AD_RA_7B_TIME  =        [1.50,  4.99,  8.49,  16.67, 17.22,  14.50, 14.31, 19.46,  19.78, 25.08, 36.12]
AD_RA_7B_hal   =        [00.00, 00.00, 00.00, 00.00, 00.00,  00.00, 00.00, 00.00,  0.03,  0.15,  0.51]





standard_RA_13B =        [36.29, 36.48, 37.00, 37.08, 38.05, 38.48, 37.55, 36.76, 23.88, 6.83, 1.57 ]
standard_RA_13B_TIME  =  [2.1,   4.5,   7.73,  11.22, 15.6,  20.3,  25.53, 31.37, 37.85, 51.4, 67.13]
standard_RA_13B_hal   =  [0,     0.0,   0.0,   0.0,   0.0 ,  0.0 ,  00.15, 00.78, 26.94, 81.93,95.67]

AD_RA_13B       =        [00.00, 00.00, 00.00, 00.00, 00.00,  00.00, 00.00, 00.00,  00.00, 00.00, 00.00]
AD_RA_13B_TIME  =        [00.00, 00.00, 00.00, 00.00, 00.00,  00.00, 00.00, 00.00,  00.00, 00.00, 00.00]
AD_RA_13B_hal   =        [00.00, 00.00, 00.00, 00.00, 00.00,  00.00, 00.00, 00.00,  00.00, 00.00, 00.00]




standard_RA_70B =        [44.78, 45.40, 46.11, 46.19, 48.23, 48.47, 48.69, 47.45, 40.53, 22.39, 7.15]
standard_RA_70B_TIME  =  [6.92,  16.17, 27.98, 40.0,  57.23, 73.35, 91.45, 111.5, 192.4, 176.43,229.77]
standard_RA_13B_hal   =  [0, 0.0,  0.0,    0.0,   0.0 ,   0.0 ,  0.0,  00.07, 1.09,  27.57, 78.71 ]

AD_RA_70B       =        [00.00, 00.00, 00.00, 00.00, 00.00,  00.00, 00.00, 00.00,  00.00, 00.00, 00.00]
AD_RA_70B_TIME  =        [00.00, 00.00, 00.00, 00.00, 00.00,  00.00, 00.00, 00.00,  00.00, 00.00, 00.00]
AD_RA_70B_hal   =        [00.00, 00.00, 00.00, 00.00, 00.00,  00.00, 00.00, 00.00,  00.00, 00.00, 00.00]





def col_1_1():
    bar1 = standard_RA_7B_ACC
    bar2 = AD_RA_7B_ACC

    # number of hops of EntailmentBank
    fig, ax1 = plt.subplots(1,1,figsize=(8, 3.6),dpi=300)
    plt.subplots_adjust(left=0.07, right=0.993, top=0.98, bottom=0.135)

    plt.grid(zorder=-100)

    bar_width = 0.35
    tick_label = num
    x = np.array(num)

    save_file = "7b-1-1.jpg" 
    ax1_xlabel = "The number of retrieval documents"
    ax1_ylabel = "Performance (%)"

    ax1.set_ylim(25,60)

    rects1 = ax1.bar(x-bar_width/2, bar1, bar_width, align="center", color='lightsteelblue', edgecolor='black', zorder=100, hatch='')
    rects2 = ax1.bar(x+bar_width/2, bar2, bar_width, align="center", color='coral', edgecolor='black', zorder=100, hatch='//')

    ax1.set_ylabel(ax1_ylabel,size=14)
    plt.yticks(size=11)

    plt.xticks(x,tick_label,size=12)
    plt.yticks(size=11)
    ax1.set_xlabel(ax1_xlabel,size=14)
    plt.legend(['  Standard-RA','  AD-RA'], fontsize=10)

    # Add text for each bar (optional: adjust the y delta to fit your needs)
    y_delta = 0.5
    min_height = ax1.get_ylim()[0] + y_delta -0.5 # set a minimum height for text

    for rect in rects1:
        height = max(rect.get_height(), min_height)
        ax1.text(rect.get_x() + rect.get_width()/2. -0.03 , height + 0.05 , f'{rect.get_height():.2f}' , ha='center', va='bottom', fontsize=6)

    for rect in rects2:
        height = max(rect.get_height(), min_height)
        ax1.text(rect.get_x() + rect.get_width()/2. +0.03 , height + 0.05 , f'{rect.get_height():.2f}' , ha='center', va='bottom', fontsize=6)

    plt.savefig(save_file)


def col_2_1():
    line1 = standard_RA_7B_TIME
    line2 = AD_RA_7B_TIME
    bar1= standard_RA_len
    bar2= AD_RA_len


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


    ax1.set_ylim(0, 5000)


    ax1.bar([i-bar_width/2 for i in x], bar1, bar_width, align="center", color='lightsteelblue', edgecolor='black', zorder=100, hatch='')
    ax1.bar([i+bar_width/2 for i in x], bar2, bar_width, align="center", color='coral', edgecolor='black', zorder=100, hatch='//')

    ax1.set_ylabel(ax1_ylabel,size=14)
    ax1.set_xlabel(ax1_xlabel,size=14)
    plt.yticks(size=11)
    # plt.legend(['  Standard-RA','  AD-RA'], fontsize=10, loc='upper left')
    plt.legend(['  Standard-RA','  AD-RA'], fontsize=10, loc=(0.01, 0.84))

    # ====================================================================================================================
    ax2 = ax1.twinx()
    ax2_ylabel = "Run time (mins.)"
    ax2.set_ylim(0, 50)


    ax2.plot(x, line1, color="green",marker="o",linestyle='-.')
    ax2.plot(x, line2, color="red",marker="s",linestyle='--')
    ax2.set_ylabel(ax2_ylabel,size=14)
    

    plt.xticks(x,tick_label,size=12)
    plt.yticks(size=11)
    
    plt.legend(['  Standard-RA','  AD-RA'], fontsize=10,  loc=(0.01, 0.67))
    plt.savefig(save_file)


def col_3_1():
    num = [6, 7, 8, 9, 10]
    bar1 = standard_RA_7B_hal[num[0]:]
    bar2 = AD_RA_7B_hal[num[0]:]

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3.6), dpi=300)
    plt.subplots_adjust(left= 0.135, right=0.993, top=0.98, bottom=0.135)

    plt.grid(zorder=-100)

    bar_width = 0.35
    tick_label = num
    x = num

    save_file = "7b-3-1.jpg"
    ax1_xlabel = "The number of retrieval documents"
    ax1_ylabel = "The ratio of hallucination (%)"

    m_v = 5.0

    ax1.set_ylim(-m_v, 102)

    rects1 = ax1.bar([i - bar_width / 2 for i in x], [i+m_v for i in bar1], bar_width, align="center", color='lightsteelblue', edgecolor='black', zorder=100, bottom=-m_v, hatch='')
    rects2 = ax1.bar([i + bar_width / 2 for i in x], [i+m_v for i in bar2], bar_width, align="center", color='coral', edgecolor='black', zorder=100, bottom=-m_v, hatch='//')

    plt.yticks(size=11)
    plt.xticks(x, tick_label, size=12)
    plt.yticks(size=11)
    ax1.set_xlabel(ax1_xlabel, size=14)
    ax1.set_ylabel(ax1_ylabel, size=14)
    plt.legend(['Standard-RA', 'AD-RA'], fontsize=10)

    # Add text for each bar
    y_delta = 0.5
    min_height = ax1.get_ylim()[0] + y_delta  # set a minimum height for text

    for rect in rects1:
        height = max(rect.get_height(), min_height) 
        ax1.text(rect.get_x() + rect.get_width() / 2. -0.01, height-m_v, str(round(float('%.2f' % rect.get_height())-m_v, 2)) , ha='center', va='bottom', fontsize=8)

    for rect in rects2:
        height = max(rect.get_height(), min_height)
        ax1.text(rect.get_x() + rect.get_width() / 2. +0.01, height-m_v, str(round(float('%.2f' % rect.get_height())-m_v, 2)) , ha='center', va='bottom', fontsize=8)

    plt.savefig(save_file)




col_1_1()
col_2_1()
col_3_1()