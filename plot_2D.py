"""
    2D plotting funtions
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import argparse
import numpy as np
from os.path import exists
import seaborn as sns


def plot_2d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(surf_file, 'r')
    # Explicitly read data into NumPy arrays
    x = f['xcoordinates'][:]
    y = f['ycoordinates'][:]
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = f[surf_name][:]
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        # Assuming surf_name was test_acc or train_acc if _err is requested
        acc_surf_name = surf_name.replace('_err', '_acc')
        if acc_surf_name in f.keys():
            Z = 100 - f[acc_surf_name][:]
        else:
            print ('%s or corresponding _acc key is not found in %s' % (surf_name, surf_file))
            f.close()
            return
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))
        f.close()
        return

    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        f.close()
        return

    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    output_filename_contour = surf_file.replace('.h5', '') + '_' + surf_name + '_2dcontour.pdf'
    fig.savefig(output_filename_contour, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_filename_contour}")

    fig = plt.figure()
    CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    output_filename_contourf = surf_file.replace('.h5', '') + '_' + surf_name + '_2dcontourf.pdf'
    fig.savefig(output_filename_contourf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_filename_contourf}")

    fig = plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    output_filename_heat = surf_file.replace('.h5', '') + '_' + surf_name + '_2dheat.pdf'
    sns_plot.get_figure().savefig(output_filename_heat, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_filename_heat}")

    fig = plt.figure()
    try:
        ax = fig.add_subplot(111, projection='3d') # More compatible way for 3D subplot
    except AttributeError: # Fallback for older matplotlib if add_subplot with projection fails
        ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    output_filename_3d = surf_file.replace('.h5', '') + '_' + surf_name + '_3dsurface.pdf'
    fig.savefig(output_filename_3d, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_filename_3d}")

    f.close()
    if show: plt.show()


def plot_trajectory(proj_file, dir_file, show=False):
    """ Plot optimization trajectory on the plane spanned by given directions."""
    assert exists(proj_file), 'Projection file does not exist: %s' % proj_file
    
    pf = None # Initialize to ensure it's defined for finally block
    f2 = None
    try:
        pf = h5py.File(proj_file, 'r')
        
        # FIX: Explicitly read HDF5 datasets into NumPy arrays before plotting
        proj_xcoord_data = pf['proj_xcoord'][:] 
        proj_ycoord_data = pf['proj_ycoord'][:]
        
        fig = plt.figure()
        plt.plot(proj_xcoord_data, proj_ycoord_data, marker='.')
        plt.tick_params('y', labelsize='x-large')
        plt.tick_params('x', labelsize='x-large')
        
        if dir_file and exists(dir_file): # Check if dir_file is provided and exists
            f2 = h5py.File(dir_file,'r')
            if 'explained_variance_ratio_' in f2.keys():
                ratio_x = f2['explained_variance_ratio_'][0]
                ratio_y = f2['explained_variance_ratio_'][1]
                plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
                plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
        else:
            print(f"Warning: dir_file '{dir_file}' not provided or does not exist. PCA info will not be plotted.")

        output_filename = proj_file.replace('.h5', '') + '_trajectory.pdf'
        fig.savefig(output_filename, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Saved: {output_filename}")
        if show: plt.show()

    except Exception as e:
        print(f"An error occurred in plot_trajectory: {e}")
    finally:
        if pf:
            pf.close()
        if f2:
            f2.close()


def plot_contour_trajectory(surf_file, dir_file, proj_file, surf_name='train_loss',
                            vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """2D contour + trajectory"""
    assert exists(surf_file), f"Surface file does not exist: {surf_file}"
    assert exists(proj_file), f"Projection file does not exist: {proj_file}"
    assert exists(dir_file), f"Direction file does not exist: {dir_file}"

    f_surf = None
    f_proj = None
    f_dir = None
    try:
        f_surf = h5py.File(surf_file,'r')
        # Explicitly read data into NumPy arrays
        x = f_surf['xcoordinates'][:]
        y = f_surf['ycoordinates'][:]
        X, Y = np.meshgrid(x, y)
        if surf_name in f_surf.keys():
            Z = f_surf[surf_name][:]
        else:
            print(f"Surface name {surf_name} not found in {surf_file}")
            return
        
        fig = plt.figure()
        CS1 = plt.contour(X, Y, Z, levels=np.arange(vmin, vmax, vlevel))
        
        # Robust logspace contour levels
        min_Z_positive = np.min(Z[Z > 0]) if np.any(Z > 0) else 0.01 # Avoid log(0)
        max_Z = np.max(Z)
        if max_Z > min_Z_positive:
             log_levels = np.logspace(np.log10(min_Z_positive), np.log10(max_Z), num=8)
             CS2 = plt.contour(X, Y, Z, levels=log_levels, alpha=0.7) # Added alpha for potentially busy plots
             plt.clabel(CS2, inline=1, fontsize=6)

        f_proj = h5py.File(proj_file, 'r')
        # FIX: Explicitly read HDF5 datasets into NumPy arrays before plotting
        proj_xcoord_data = f_proj['proj_xcoord'][:]
        proj_ycoord_data = f_proj['proj_ycoord'][:]
        plt.plot(proj_xcoord_data, proj_ycoord_data, marker='.')

        f_dir = h5py.File(dir_file,'r')
        if 'explained_variance_ratio_' in f_dir.keys():
            ratio_x = f_dir['explained_variance_ratio_'][0]
            ratio_y = f_dir['explained_variance_ratio_'][1]
            plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
            plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
        
        plt.clabel(CS1, inline=1, fontsize=6)
        
        output_filename = proj_file.replace('.h5', '') + '_' + surf_name + '_contour_trajectory.pdf'
        fig.savefig(output_filename, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Saved: {output_filename}")
        
        if show: plt.show()

    except Exception as e:
        print(f"An error occurred in plot_contour_trajectory: {e}")
    finally:
        if f_surf:
            f_surf.close()
        if f_proj:
            f_proj.close()
        if f_dir:
            f_dir.close()


def plot_2d_eig_ratio(surf_file, val_1='min_eig', val_2='max_eig', show=False):
    """ Plot the heatmap of eigenvalue ratios, i.e., |min_eig/max_eig| of hessian """
    print('------------------------------------------------------------------')
    print('plot_2d_eig_ratio')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    
    f = None
    try:
        f = h5py.File(surf_file,'r')
        if val_1 not in f.keys() or val_2 not in f.keys():
            print(f"Eigenvalue keys {val_1} or {val_2} not found in {surf_file}")
            return

        Z1 = f[val_1][:]
        Z2 = f[val_2][:]
    except Exception as e:
        print(f"Error reading eigenvalue data from {surf_file}: {e}")
        return
    finally:
        if f:
            f.close()

    epsilon = 1e-9 # Adjusted epsilon
    
    # Calculate absolute ratio
    abs_ratio = np.divide(np.absolute(Z1), np.absolute(Z2) + epsilon, 
                          out=np.zeros_like(Z1, dtype=float), 
                          where=(np.absolute(Z2) + epsilon)!=0) # Robust division
    
    fig_abs = plt.figure()
    # Use percentile for vmax to handle outliers better
    vmax_abs = np.percentile(abs_ratio[np.isfinite(abs_ratio)], 99) if np.any(np.isfinite(abs_ratio)) else 1.0
    sns_plot_abs = sns.heatmap(abs_ratio, cmap='viridis', vmin=0, vmax=vmax_abs, cbar=True,
                               xticklabels=False, yticklabels=False)
    sns_plot_abs.invert_yaxis()
    output_filename_abs_ratio = surf_file.replace('.h5', '') + '_' + val_1 + '_' + val_2 + '_abs_ratio_heat_sns.pdf'
    sns_plot_abs.get_figure().savefig(output_filename_abs_ratio, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_filename_abs_ratio}")

    # Calculate direct ratio
    ratio = np.divide(Z1, Z2 + epsilon, 
                      out=np.zeros_like(Z1, dtype=float), 
                      where=(Z2 + epsilon)!=0) # Robust division

    fig_ratio = plt.figure()
    finite_ratio = ratio[np.isfinite(ratio)]
    robust_min = np.percentile(finite_ratio, 1) if len(finite_ratio) > 0 else -1.0
    robust_max = np.percentile(finite_ratio, 99) if len(finite_ratio) > 0 else 1.0
    if robust_min == robust_max: # Handle case where all values are the same or very few distinct values
        robust_min -= 0.5
        robust_max += 0.5
        if robust_min == robust_max: # Still same, e.g. all zeros
            robust_max = robust_min + 1.0


    sns_plot_ratio = sns.heatmap(ratio, cmap='viridis', vmin=robust_min, vmax=robust_max, cbar=True, 
                                 xticklabels=False, yticklabels=False)
    sns_plot_ratio.invert_yaxis()
    output_filename_ratio = surf_file.replace('.h5', '') + '_' + val_1 + '_' + val_2 + '_ratio_heat_sns.pdf'
    sns_plot_ratio.get_figure().savefig(output_filename_ratio, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_filename_ratio}")
    
    if show: plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 2D loss surface')
    parser.add_argument('--surf_file', '-f', default='', help='The h5 file that contains surface values')
    parser.add_argument('--dir_file', default='', help='The h5 file that contains directions (for PCA info)')
    parser.add_argument('--proj_file', default='', help='The h5 file that contains the projected trajectories')
    parser.add_argument('--surf_name', default='train_loss', help='The type of surface to plot (e.g., train_loss, test_loss, min_eig, max_eig)')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map for contour plots')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map for contour plots')
    parser.add_argument('--vlevel', default=0.5, type=float, help='Plot contours every vlevel for contour plots')
    parser.add_argument('--show', action='store_true', default=False, help='show plots')

    args = parser.parse_args()

    if args.proj_file and exists(args.proj_file):
        if args.surf_file and exists(args.surf_file) and args.dir_file and exists(args.dir_file):
            print(f"Plotting contour trajectory: surf_file={args.surf_file}, dir_file={args.dir_file}, proj_file={args.proj_file}")
            plot_contour_trajectory(args.surf_file, args.dir_file, args.proj_file,
                                    args.surf_name, args.vmin, args.vmax, args.vlevel, args.show)
        else:
            print(f"Plotting trajectory: proj_file={args.proj_file}, dir_file (optional for PCA info)={args.dir_file if args.dir_file and exists(args.dir_file) else 'Not provided or does not exist'}")
            plot_trajectory(args.proj_file, args.dir_file if args.dir_file and exists(args.dir_file) else "", args.show)
    elif args.surf_file and exists(args.surf_file):
        print(f"Plotting 2D data from: surf_file={args.surf_file}")
        if 'eig' in args.surf_name.lower(): # A simple check if it's an eigenvalue related plot
             print("Attempting to plot eigenvalue ratios. Assuming 'min_eig' and 'max_eig' keys exist in the HDF5 file.")
             plot_2d_eig_ratio(args.surf_file, 'min_eig', 'max_eig', args.show)
        else:
            plot_2d_contour(args.surf_file, args.surf_name, args.vmin, args.vmax, args.vlevel, args.show)
    else:
        print("No valid file provided for plotting. Please specify --proj_file (for trajectories) or --surf_file (for surfaces/contours).")