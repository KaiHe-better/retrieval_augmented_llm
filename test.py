
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