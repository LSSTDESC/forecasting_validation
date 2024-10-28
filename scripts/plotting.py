import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr


def get_colors(data, cmap="cmr.pride", cmap_range=(0.15, 0.85)):
    num = len(data)

    colors = cmr.take_cmap_colors(cmap, num, cmap_range=cmap_range, return_fmt='hex')

    return colors


def compare_two_data_vector_sets_absolute(data_vector_1,
                                          data_vector_2,
                                          ells_1,
                                          ells_2,
                                          cmap_1="cmr.pride",
                                          cmap_2="viridis", ):
    colors_1 = get_colors(data_vector_1, cmap=cmap_1, cmap_range=(0.15, 0.85))
    colors_2 = get_colors(data_vector_2, cmap=cmap_2, cmap_range=(0.15, 0.85))

    for i in range(data_vector_1.shape[1]):  # Loop over 15 cls
        plt.loglog(ells_1, data_vector_1[:, i], c=colors_1[i])
        plt.loglog(ells_2, data_vector_2[:, i], c=colors_2[i], ls=":", lw=3)

    plt.xlabel("Multipole Scale $\\ell$", fontsize=18)
    plt.ylabel("Angular Power Spectrum $C_{\\ell}$", fontsize=18)
    # plt.legend()
    plt.show()


def compare_two_data_vector_sets_relative(data_vector_1,
                                          data_vector_2,
                                          ells_1,
                                          ells_2,
                                          cmap_1="cmr.pride",
                                          cmap_2="cmr.pride",
                                          label_1="X",
                                          label_2="Y"
                                          ):
    colors_1 = get_colors(data_vector_1, cmap=cmap_1, cmap_range=(0.15, 0.85))
    colors_2 = get_colors(data_vector_2, cmap=cmap_2, cmap_range=(0.15, 0.85))

    for i in range(data_vector_1.shape[1]):  # Loop over 15 cls
        plt.plot(ells_1, data_vector_1[:, i] / data_vector_2[:, i] - 1, c=colors_1[3])
        plt.axhline(0, c="gray")

    plt.plot([], [], c="white", label=f"X = {label_1} \n Y = {label_2}")

    plt.xlabel("Multipole Scale $\\ell$", fontsize=18)
    plt.ylabel("$C_{\\ell}^X / C_{\\ell}^Y - 1 $", fontsize=18)
    plt.legend(fontsize=18, frameon=False)
    plt.show()
