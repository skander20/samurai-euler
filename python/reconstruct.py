import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import argparse


def pred_coeff(pred_s, sign):
    if pred_s == 0:
        return np.array([1])
    elif pred_s == 1:
        return np.array([sign / 8, 1, -sign / 8])
    elif pred_s == 2:
        return np.array(
            [
                -sign * 3.0 / 128.0,
                sign * 22.0 / 128.0,
                1,
                -sign * 22 / 128.0,
                sign * 3.0 / 128.0,
            ]
        )


def read_frame_euler_2d(filename):
    mesh = h5py.File(filename + ".h5", "r")["mesh"]
    points = mesh["points"]
    connectivity = mesh["connectivity"]

    xyz = points[:][connectivity[:]][:, :, :]
    centers = 0.25 * (xyz[:, 0, :] + xyz[:, 1, :] + xyz[:, 2, :] + xyz[:, 3, :])

    sol = {}
    sol["rho"] = mesh["fields"]["rho"][:]
    sol["pressure"] = mesh["fields"]["pressure"][:]
    sol["ux"] = mesh["fields"]["velocity_0"][:]
    sol["uy"] = mesh["fields"]["velocity_1"][:]
    levels = mesh["fields"]["levels"][:]

    return centers[:, 0], centers[:, 1], sol, levels


def double_mach_reflection_bc(Tf, pred_s, level, xmin, u, var_name):
    alpha = np.pi / 3.0
    x0 = 2.0 / 3

    left_state = {
        "rho": 8.0,
        "pressure": 116.5,
        "ux": 8.25 * np.sin(alpha),
        "uy": -8.25 * np.cos(alpha),
    }

    right_state = {"rho": 1.4, "pressure": 1.0, "ux": 0.0, "uy": 0.0}

    # left boundary
    for g in range(pred_s):
        u[g, :] = left_state[var_name]
    # right boundary
    for g in range(pred_s):
        u[-(g + 1), :] = right_state[var_name]
    # top boundary
    dx = 2 ** (-level)
    for i in range(4 * 2**level):
        x = xmin + (i + 0.5) * dx
        x1 = x0 + 10 * Tf / np.sin(alpha) + 1 / np.tan(alpha)
        if x < x1:
            for g in range(pred_s):
                u[i + pred_s, -(g + 1)] = left_state[var_name]
        else:
            for g in range(pred_s):
                u[i + pred_s, -(g + 1)] = right_state[var_name]
    # bottom boundary
    for i in range(4 * 2**level):
        x = xmin + (i + 0.5) * dx
        if x < x0:
            for g in range(pred_s):
                u[i + pred_s, g] = left_state[var_name]
        else:
            for g in range(pred_s):
                u[i + pred_s, g] = right_state[var_name]


def recons_2d(box, x, y, pred_s, Tf, sol, level, var_name="rho"):
    xmin, ymin = box[0]
    xmax, ymax = box[1]
    xsize = int(xmax - xmin)
    ysize = int(ymax - ymin)
    dx_func = lambda level: min(xsize, ysize) * 2 ** (-level)

    min_level = int(np.min(level))
    max_level = int(np.max(level))

    u = sol[var_name]

    # read leaf cells
    ul = {}
    for ilevel in range(min_level, max_level + 1):
        dx = dx_func(ilevel)
        ul[ilevel] = np.empty(
            (xsize * 2**ilevel + 2 * pred_s, ysize * 2**ilevel + 2 * pred_s)
        )
        ul[ilevel][:] = np.nan
        (index,) = np.where(level == ilevel)

        index_x = ((x[index] - xmin - 0.5 * dx) / dx).astype(int)
        index_y = ((y[index] - ymin - 0.5 * dx) / dx).astype(int)
        ul[ilevel][index_x + pred_s, index_y + pred_s] = u[index]

    for ilevel in range(min_level, max_level + 1):
        double_mach_reflection_bc(Tf, pred_s, ilevel, xmin, ul[ilevel], var_name)

    # projection of leaves (vectorized, stride 2)
    for ilevel in range(max_level - 1, min_level - 1, -1):
        parent = ul[ilevel]
        child = ul[ilevel + 1]
        child_interior = child[pred_s:-pred_s, pred_s:-pred_s]
        # Extract all 2x2 non-overlapping patches from child (stride 2)
        patches = sliding_window_view(child_interior, (2, 2))[::2, ::2]
        patch_sums = np.sum(patches, axis=(2, 3))
        # Assign directly to parent interior
        parent_interior = parent[pred_s:-pred_s, pred_s:-pred_s]
        mask = ~np.isnan(child_interior[::2, ::2])
        parent_interior[mask] = 0.25 * patch_sums[mask]

    # Compute the four stencils
    st00 = np.outer(pred_coeff(pred_s, 1), pred_coeff(pred_s, 1))
    st10 = np.outer(pred_coeff(pred_s, -1), pred_coeff(pred_s, 1))
    st01 = np.outer(pred_coeff(pred_s, 1), pred_coeff(pred_s, -1))
    st11 = np.outer(pred_coeff(pred_s, -1), pred_coeff(pred_s, -1))

    # prediction (vectorized)
    for ilevel in range(min_level, max_level):
        parent = ul[ilevel]
        child = ul[ilevel + 1]

        # Extract all (2*pred_s+1, 2*pred_s+1) patches
        patches = sliding_window_view(parent, (2 * pred_s + 1, 2 * pred_s + 1))

        # Compute the four child values (vectorized sum over last two dims)
        vals00 = np.tensordot(patches, st00, axes=([2, 3], [0, 1]))
        vals10 = np.tensordot(patches, st10, axes=([2, 3], [0, 1]))
        vals01 = np.tensordot(patches, st01, axes=([2, 3], [0, 1]))
        vals11 = np.tensordot(patches, st11, axes=([2, 3], [0, 1]))

        child_interior = child[pred_s:-pred_s, pred_s:-pred_s]
        mask = np.isnan(child_interior[::2, ::2])

        child_interior[::2, ::2][mask] = vals00[mask]
        child_interior[1::2, ::2][mask] = vals00[mask]
        child_interior[::2, 1::2][mask] = vals10[mask]
        child_interior[1::2, 1::2][mask] = vals11[mask]

    if pred_s > 0:
        return ul[max_level][pred_s:-pred_s, pred_s:-pred_s]
    return ul[max_level][:]


def main():
    parser = argparse.ArgumentParser(description="Samurai Euler Reconstruction")
    parser.add_argument(
        "--pred_s", type=int, default=1, help="Prediction stencil order (0, 1, 2)"
    )
    parser.add_argument("--Tf", type=float, default=0.2, help="Final time Tf")
    parser.add_argument(
        "--filename",
        type=str,
        default="../build/results/double_mach_reflection_hllc_ite_9",
        help="Input filename (without .h5)",
    )
    args = parser.parse_args()

    filename = args.filename
    box = [[0.0, 0.0], [4.0, 1.0]]

    x, y, sol, level = read_frame_euler_2d(filename)
    r = recons_2d(box, x, y, args.pred_s, args.Tf, sol, level, var_name="rho")

    import matplotlib.pyplot as plt

    plt.imshow(r.T, origin="lower")
    plt.show()


if __name__ == "__main__":
    main()
