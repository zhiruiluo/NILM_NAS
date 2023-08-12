import matplotlib.pyplot as plt
import numpy as np

def get_pareto_front(Xs, Ys, maxX=True, maxY=True):
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    return pareto_front

def add_mid_point_pareto_front(pareto_front: list, maxX=True, maxY=True):
    new_pareto_front =[]
    for i in range(len(pareto_front)-1):
        a = pareto_front[i] 
        b = pareto_front[i+1]
        new_pareto_front.append(a)
        if not maxX and maxY:
            new_pareto_front.append([b[0],a[1]])
        elif not maxX and not maxY:
            new_pareto_front.append([b[0],a[1]])
            
    new_pareto_front.append(pareto_front[-1])
    return new_pareto_front

def get_hypervolume_front(Xs, Ys, reference_point, maxX=True, maxY=True):
    pareto_front = get_pareto_front(Xs, Ys, maxX, maxY)
    pareto_front = add_mid_point_pareto_front(pareto_front, maxX, maxY)
    if not maxX and maxY:
        pareto_front = [[pareto_front[0][0], reference_point[1]]] + pareto_front
        pareto_front = pareto_front + [[reference_point[0], pareto_front[-1][1]]]
    return pareto_front

def plot_hypervolume(x, y, maxX=False, maxY=False):
    pareto_front = get_pareto_front(x, y, maxX, maxY)
    pareto_front_mid = add_mid_point_pareto_front(pareto_front,maxX, maxY)
    
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    
    pf_X_mid = [pair[0] for pair in pareto_front_mid]
    pf_Y_mid = [pair[1] for pair in pareto_front_mid]
    
    nadir_point = [max(pf_X), max(pf_Y)]
    reference_point = (max(pf_X)+0.5, max(pf_Y)+0.5)
    pf_X_mid = [pf_X_mid[0]] + pf_X_mid + [reference_point[0]]
    pf_Y_mid = [reference_point[1]] + pf_Y_mid + [pf_Y_mid[-1]]
    
    fig, ax = plt.subplots()
    ax.plot(pf_X, pf_Y,  marker='o', markersize=4, c='#4270b6', linestyle='',mfc='black', mec='black', zorder=3)
    ax.plot(pf_X_mid, pf_Y_mid, linestyle='-', markersize=4, c='#4270b6', mfc='black', mec='black',zorder=2)
    
    
    
    ax.annotate('Reference point', xy=(reference_point[0], reference_point[1]), xycoords='data', xytext=(
        0.8, 1.13), textcoords='axes fraction', va='top', ha='left', arrowprops=dict(facecolor='black', shrink=0.10, width=1,headwidth=8,headlength=8),fontsize=14)
    ax.plot(reference_point[0], reference_point[1],
             'o', color='red', markersize=8)
    
    ax.annotate('Nadir point', xy=nadir_point, xycoords='data', xytext=(nadir_point[0],nadir_point[1]+0.3), 
                textcoords='data', va='top', ha='center', arrowprops=dict(facecolor='black', shrink=0.10, width=1,headwidth=8,headlength=8),fontsize=14)
    ax.plot(nadir_point[0], nadir_point[1], 'o', color='blue', markersize=8)
    ax.plot((pareto_front[0][0], nadir_point[0]),(pareto_front[0][1],nadir_point[1]), linestyle='--', c='#4270b6')
    ax.plot((pareto_front[-1][0], nadir_point[0]),(pareto_front[-1][1],nadir_point[1]), linestyle='--', c='#4270b6')
    
    ax.annotate('The hypervolume', xy=(3.5, 3.5), xycoords='data',fontsize=14)
    ys = [reference_point[1]] + pf_Y_mid
    xs = [pf_X_mid[0]] + pf_X_mid
    ax.fill_betweenx(ys, xs, reference_point[0] * np.ones(len(xs)),
                      color='#dfeaff', alpha=1, zorder=1)

    pareto_front = np.array(pareto_front)
    pareto_front_mid = np.array(pareto_front_mid)

    # plot pareto front
    ax.plot(pf_X, pf_Y, color='red', linewidth=2, linestyle='-')
    ax.annotate('Pareto Front', xy=(pareto_front_mid[-16][0],pareto_front_mid[-16][1]), xycoords='data', xytext=(0.02,0.4), textcoords='axes fraction', arrowprops=dict(color='black', shrink=0.20, width=1,headwidth=8,headlength=8),fontsize=14)

    ax.annotate('A solution',fontsize=12, xy=pareto_front[-1], xycoords='data', xytext=pareto_front[-1]-np.array([0.7,0.2]), textcoords='data',va='bottom', ha='left', arrowprops=dict(facecolor='black', shrink=0.10, width=1,headwidth=8,headlength=8))
    # ax.annotate('Attainment surface',fontsize=12, xy=(pareto_front_mid[-7][1]-0.05,pareto_front_mid[-6][0]-0.05), xycoords='data', xytext=(0.1,0.1), textcoords='axes fraction',va='bottom', ha='left', arrowprops=dict(facecolor='black', shrink=0.03, width=1,headwidth=8,headlength=8))
    ax.annotate('Attainment surface', xy=(pareto_front_mid[-10][1]-0.05,pareto_front_mid[-9][0]-0.05), xycoords='data', xytext=(0.1,0.18), textcoords='axes fraction',va='bottom', ha='left', arrowprops=dict(facecolor='black', shrink=0.03, width=1,headwidth=8,headlength=8),fontsize=14)
    ax.set_ylim(2.2,4.5)
    ax.set_xlim(2.2,4.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    arrow_style = "->"  # or "<-", "<->", etc.
    # ax.plot((0,1), (0,0), marker=">", ms=10, color="k", linestyle='-')
    # ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
    #         clip_on=False)
    # Optional: Set the visible property of the spines
    # to False if you don't want to show the default
    # spines along with the arrows
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.xlabel(r"Objective 1")
    plt.ylabel(r"Objective 2")
    path = 'results/hypervolume.pdf'
    fig.savefig(path)
    print('save path', path)
    
    
np.random.seed(2)
x = np.arange(2.5,4, 0.1)
y = -x + np.random.uniform(-1,1,len(x)) * 0.2 + 6.5

plot_hypervolume(x,y)