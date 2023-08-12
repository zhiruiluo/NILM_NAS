
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])

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
            new_pareto_front.append([a[0],b[1]])
    new_pareto_front.append(pareto_front[-1])
    return new_pareto_front


def plot_hyper_volume(x, y):
    x = np.array(x)
    # y = np.array(y)

    # # Zip x and y into a numpy ndarray
    # coordinates = np.array(sorted(zip(x, y)))

    # # Empty pareto set
    # pareto_set = np.full(coordinates.shape, np.inf)

    # i = 0
    # for point in coordinates:
    #     if i == 0:
    #         pareto_set[i] = point
    #         i += 1
    #     elif point[1] < pareto_set[:, 1].min():
    #         pareto_set[i] = point
    #         i += 1

    # # Get rid of unused spaces
    # pareto_set = pareto_set[:i, :]
    
    pareto_set = get_pareto_front(x, y, maxX=False, maxY=False)
    pareto_set_mid = add_mid_point_pareto_front(pareto_set, maxX=False, maxY=False)
    print(pareto_set)
    print(pareto_set_mid)

    pf_X = [pair[0] for pair in pareto_set]
    pf_Y = [pair[1] for pair in pareto_set]
    
    pf_X_mid = [pair[0] for pair in pareto_set_mid]
    pf_Y_mid = [pair[1] for pair in pareto_set_mid]

    pareto_set = np.array(pareto_set)
    pareto_set_mid = np.array(pareto_set_mid)

    # Add reference point to the pareto set
    # pareto_set[i]
    reference_point = (np.max(pf_X),np.max(pf_Y))
    print(reference_point)

    # These points will define the path to be plotted and filled
    x_path_of_points = []
    y_path_of_points = []

    plt.figure(figsize=(4, 4))

    for index, point in enumerate(pareto_set):

        if index < len(pareto_set) - 1:
            # plt.plot([point[0], point[0]], [point[1], pareto_set[index + 1][1]], marker='o', markersize=4, c='#4270b6', linestyle='-',
            #          mfc='black', mec='black')
            # plt.plot([point[0], pareto_set[index + 1][0]], [pareto_set[index + 1][1], pareto_set[index + 1][1]], linestyle='-',
            #          marker='o', markersize=4, c='#4270b6', mfc='black', mec='black')

            x_path_of_points += [point[0], point[0], pareto_set[index + 1][0]]
            y_path_of_points += [point[1], pareto_set[index + 1]
                                 [1], pareto_set[index + 1][1]]
    
    plt.plot(pf_X, pf_Y,  marker='o', markersize=4, c='#4270b6', linestyle='',mfc='black', mec='black', zorder=2)
    plt.plot(pf_X_mid, pf_Y_mid, linestyle='-', markersize=4, c='#4270b6', mfc='black', mec='black',zorder=2)
    
    print(pareto_set)
    # Link 1 to Reference Point
    plt.plot([pareto_set[0][0], reference_point[0]], [pareto_set[0][1], reference_point[1]], marker='o', markersize=4, linestyle='-',
             c='#4270b6', mfc='black', mec='black')
    # Link 2 to Reference Point
    plt.plot([pareto_set[-1][0], reference_point[0]], [pareto_set[-1][1], reference_point[1]], marker='o', markersize=4, linestyle='-',
             c='#4270b6', mfc='black', mec='black')
    # Highlight the Reference Point
    plt.annotate('Nadir point', xy=(reference_point[0], reference_point[1]), xycoords='data', xytext=(
        0.95, 1.1), textcoords='axes fraction', va='top', ha='left', arrowprops=dict(facecolor='black', shrink=0.10, lw=0.5))
    plt.plot(reference_point[0], reference_point[1],
             'o', color='red', markersize=8)

    # Fill the area between the Pareto set and Ref y
    plt.annotate('The hypervolume', xy=(3.2, 3.5), xycoords='data')
    plt.fill_betweenx(y_path_of_points, x_path_of_points, max(x_path_of_points) * np.ones(len(x_path_of_points)),
                      color='#dfeaff', alpha=1, zorder=1)

    # Annotation for solution
    plt.annotate('A solution', xy=pareto_set[-3], xycoords='data', xytext=pareto_set[-3]-0.2, textcoords='data',va='bottom', ha='left', arrowprops=dict(facecolor='black', shrink=0.10, width=0.5,headwidth=4,headlength=4))
    plt.annotate('Attainment surface', xy=(pareto_set[-7][1]-0.05,pareto_set[-6][0]-0.05), xycoords='data', xytext=(0.1,0.1), textcoords='axes fraction',va='bottom', ha='left', arrowprops=dict(facecolor='black', shrink=0.10, width=0.5,headwidth=4,headlength=4))

    # plt.xticks([])
    # plt.yticks([])
    plt.grid(visible=False)
    plt.xlabel(r"Objective 1")
    plt.ylabel(r"Objective 2")
    plt.tight_layout()

    plt.savefig('results/hypervolume_illustrate.pdf')


x = [ 3.253687689154591, 2.6652460044515833, 4.457925635186012, 3.2080041315472436,
     3.196941509672105, 3.1489091846746784, 3.3727329491336504,  3.330938419470396,
     3.0504985412687606, 3.9698249482752517, 3.1229599570521005, 3.1278496698518365, 
      3.1531143837247226, 3.13276172330508, 3.2136444869087857, 3.123114522218978,
    #  2.8975316624890177, 3.855654194881272, 2.982889291813081, 4.001132534228973, 3.222172022988187,
    #  3.2918121975763426, 3.119437722697402, 3.1915652020314855, 3.228511161109151, 3.410632525789594,
     3.303983909300615, 3.23152665486737, 3.12295981695552, 3.123114522218978, 3.2134999576177727, 
     3.3042667387038263, 3.379569640868453, 3.2597834943233255, 3.2365405477218783, 3.2911687133624765, 
    #  3.1835704013006616, 3.1363291696973903, 3.1422814239459718, 3.1202417240558282, 3.1311337111075535, 
     3.630375287962374, 3.181708872213033, 3.2993090610457774, 3.130988434129236, 3.12295981695552]

y = [ 3.446106995806536, 4.203717903464938, 2.53919767967234, 3.550936505497275, 
     3.553107544090778, 3.648527616343033, 3.3076630507066875,  3.320019542824605, 
     3.752735495100324, 2.853419245618656, 3.71421025754152, 3.709380479951111,  
      3.7140635724580386, 3.6981877554394647, 3.6799057516573295, 3.714041693123077, 
    #  3.831023904192475, 2.873083146950819, 4.195177914971685, 2.8302302203075165, 3.4285829711629616, 
    #  3.3624540805968035, 3.7156683374998387, 3.5623734317415163, 3.627757758118092, 3.2755855532302336,
     3.318743777730811, 3.7213811168338164, 3.714210410334474, 3.714041693123077, 3.45357840149967, 
     3.6337156456167627, 3.270784928858892, 3.400665041601096, 3.5451613263457076, 3.357372990242752, 
    #  3.5705676117602683, 3.6930983812240736, 3.687202266647831, 3.717332517575802, 3.7061199284167357, 
     3.1769420991200708, 3.492240477791187, 3.512518414215774, 3.7040103293332383, 3.714210410334474]

# x = np.random.uniform(0,1,50) + 3
# y = np.random.uniform(0,1,50) + 3
np.random.seed(2)
x = np.arange(2.5,4, 0.1)
y = -x + np.random.uniform(-1,1,len(x)) * 0.4 + 6.5


ref = [np.max(x), np.max(y)]
x = np.append(x, ref[0])
y = np.append(y, ref[1])

plot_hyper_volume(x=x, y=y)

# if __name__ == "__main__":
#     main()
