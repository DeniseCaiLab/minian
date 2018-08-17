import glob
import os
import re
import gc
import matplotlib
import pickle as pkl
import skvideo.io as sio
import xarray as xr
import numpy as np
import functools as fct
import holoviews as hv
import dask as da
from copy import deepcopy
from scipy import ndimage as ndi
from scipy.io import loadmat
from natsort import natsorted
from matplotlib import pyplot as plt
from matplotlib import animation as anim
from collections import Iterable
from tifffile import imsave, imread
from pandas import Timestamp
from IPython.core.debugger import set_trace

# import caiman as cm
# from caiman import motion_correction
# from caiman.components_evaluation import estimate_components_quality
# from caiman.miniscope.plot import plot_components
# from caiman.source_extraction import cnmf


def load_videos(vpath, pattern='msCam[0-9]+\.avi$'):
    """Load videos from a folder.

    Load videos from the folder specified in `vpath` and according to the regex
    `pattern`, then concatenate them together across time and return a
    `xarray.DataArray` representation of the concatenated videos. The default
    assumption is video filenames start with ``msCam`` followed by at least a
    number, and then followed by ``.avi``. In addition, it is assumed that the
    name of the folder correspond to a recording session identifier.

    Parameters
    ----------
    vpath : str
        The path to search for videos
    pattern : str, optional
        The pattern that describes filenames of videos. (Default value =
        'msCam[0-9]+\.avi')

    Returns
    -------
    xarray.DataArray or None
        The labeled 3-d array representation of the videos with dimensions:
        ``frame``, ``height`` and ``width``. Returns ``None`` if no data was
        found in the specified folder.

    """
    vpath = os.path.normpath(vpath)
    vlist = natsorted([
        vpath + os.sep + v for v in os.listdir(vpath) if re.search(pattern, v)
    ])
    if not vlist:
        print("No data with pattern {} found in the specified folder {}".
              format(pattern, vpath))
        return
    else:
        print("loading {} videos in folder {}".format(len(vlist), vpath))
        varray = [sio.vread(v, as_grey=True) for v in vlist]
        varray = np.squeeze(np.concatenate(varray))
        return xr.DataArray(
            varray,
            dims=['frame', 'height', 'width'],
            coords={
                'frame': range(varray.shape[0]),
                'height': range(varray.shape[1]),
                'width': range(varray.shape[2])
            },
            name=os.path.basename(vpath))


def create_fig(varlist, nrows, ncols, **kwargs):
    if not isinstance(varlist, list):
        varlist = [varlist]
    if not (nrows or ncols):
        nrows = 1
        ncols = len(varlist)
    elif nrows and not ncols:
        ncols = np.ceil(np.float(len(varlist)) / nrows).astype(int)
    elif ncols and not nrows:
        nrows = np.ceil(np.float(len(varlist)) / ncols).astype(int)
    size = kwargs.pop('size', 5)
    aspect = kwargs.pop('aspect',
                        varlist[0].sizes['width'] / varlist[0].sizes['height'])
    figsize = kwargs.pop('figsize', (aspect * size * ncols, size * nrows))
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    if not isinstance(ax, Iterable):
        ax = np.array([ax])
    return fig, ax, varlist, kwargs


def animate_video(varlist, nrows=None, ncols=None, framerate=30, **kwargs):
    fig, ax, varlist, kwargs = create_fig(varlist, nrows, ncols, **kwargs)
    frms = np.min([var.sizes['frame'] for var in varlist])
    f_update = fct.partial(
        multi_im,
        varlist,
        ax=ax,
        add_colorbar=False,
        animated=True,
        tight=False,
        **kwargs)
    f_init = fct.partial(
        multi_im, varlist, subidx={'frame': 0}, ax=ax, **kwargs)
    anm = anim.FuncAnimation(
        fig,
        func=lambda f: f_update(subidx={'frame': f}),
        init_func=f_init,
        frames=frms,
        interval=1000.0 / framerate)
    return fig, anm


def multi_im(varlist,
             subidx=None,
             ax=None,
             nrows=None,
             ncols=None,
             animated=False,
             tight=True,
             **kwargs):
    if ax is None:
        fig, ax, varlist, kwargs = create_fig(varlist, nrows, ncols, **kwargs)
    for ivar, cur_var in enumerate(varlist):
        if subidx:
            va = cur_var.loc[subidx]
        else:
            va = cur_var
        if animated:
            ax[ivar].findobj(matplotlib.collections.QuadMesh)[0].set_array(
                np.ravel(va))
            ax[ivar].set_title("frame = {}".format(int(va.coords['frame'])))
        else:
            ax[ivar].clear()
            va.plot(ax=ax[ivar], **kwargs)
    if tight:
        ax[0].get_figure().tight_layout()
    return ax


def plot_fluorescence(varlist, ax=None, nrows=None, ncols=None, **kwargs):
    if ax is None:
        fig, ax, varlist, kwargs = create_fig(varlist, nrows, ncols, **kwargs)
    for ivar, cur_var in enumerate(varlist):
        cur_mean = cur_var.mean(dim='height').mean(dim='width')
        cur_max = cur_var.max(dim='height').max(dim='width')
        cur_min = cur_var.min(dim='height').min(dim='width')
        ax[ivar].plot(cur_mean.indexes['frame'], cur_mean)
        ax[ivar].fill_between(
            cur_mean.indexes['frame'], cur_min, cur_max, alpha=0.2)
        ax[ivar].set_xlabel('frame')
        ax[ivar].set_ylabel('fluorescence')
        ax[ivar].set_title(cur_var.name)
    return ax


def save_video(movpath, fname_mov_orig, fname_mov_rig, fname_AC, fname_ACbf,
               dsratio):
    """

    Parameters
    ----------
    movpath :

    fname_mov_orig :

    fname_mov_rig :

    fname_AC :

    fname_ACbf :

    dsratio :


    Returns
    -------


    """
    mov_orig = np.load(fname_mov_orig, mmap_mode='r')
    mov_rig = np.load(fname_mov_rig, mmap_mode='r')
    mov_ac = np.load(fname_AC, mmap_mode='r')
    mov_acbf = np.load(fname_ACbf, mmap_mode='r')
    vw = sio.FFmpegWriter(
        movpath, inputdict={'-framerate': '30'}, outputdict={'-r': '30'})
    for fidx in range(0, mov_orig.shape[0], dsratio):
        print("writing frame: " + str(fidx))
        fm_orig = mov_orig[fidx, :, :] * 255
        fm_rig = mov_rig[fidx, :, :] * 255
        fm_acbf = mov_acbf[fidx, :, :] * 255
        fm_ac = mov_ac[fidx, :, :] * 255
        fm = np.concatenate(
            [
                np.concatenate([fm_orig, fm_rig], axis=1),
                np.concatenate([fm_acbf, fm_ac], axis=1)
            ],
            axis=0)
        vw.writeFrame(fm)
    vw.close()


def save_mp4(filename, dat):
    """

    Parameters
    ----------
    filename :

    dat :


    Returns
    -------


    """
    vw = sio.FFmpegWriter(
        filename,
        inputdict={'-framerate': '30'},
        outputdict={'-r': '30',
                    '-vcodec': 'rawvideo'})
    for fid, f in enumerate(dat):
        print("writing frame: {}".format(fid), end='\r')
        vw.writeFrame(f)
    vw.close()


def mov_to_uint8(mov):
    """

    Parameters
    ----------
    mov :


    Returns
    -------



    """
    return np.uint8((mov - np.min(mov)) / (np.max(mov) - np.min(mov)) * 255)


def mov_to_float32(mov):
    """

    Parameters
    ----------
    mov :


    Returns
    -------


    """
    return np.float32((mov - np.min(mov)) / (np.max(mov) - np.min(mov)))


def varr_to_uint8(varr):
    varr_max = varr.max()
    varr_min = varr.min()
    return ((varr - varr_min) / (varr_max - varr_min) * 255).astype(
        np.uint8, copy=False)


def varr_to_float32(varr):
    varr = varr.astype(np.float32, copy=False)
    varr_max = varr.max()
    varr_min = varr.min()
    varr, varr_min_bd = xr.broadcast(varr, varr_min)
    varr_norm = varr - varr_min_bd
    del varr_min_bd
    gc.collect()
    varr_norm, varr_denom = xr.broadcast(varr_norm, (varr_max - varr_min))
    varr_norm = varr_norm / varr_denom
    del varr_denom
    return varr_norm


def scale_varr(varr, scale=(0, 1), inplace=True):
    copy = not inplace
    if np.issubdtype(varr.dtype, np.floating):
        dtype = varr.dtype
    else:
        dtype = np.float32
    varr_norm = varr.astype(dtype, copy=copy)
    varr_max = varr_norm.max()
    varr_min = varr_norm.min()
    varr_norm -= varr_min
    varr_norm *= 1 / (varr_max - varr_min)
    varr_norm *= (scale[1] - scale[0])
    varr_norm += scale[0]
    return varr_norm.astype(varr.dtype, copy=False)


def varray_to_tif(filename, varr):
    imsave(filename, varr.transpose('frame', 'height', 'width'))


def tif_to_varray(filename):
    arr = imread(filename)
    f = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    varr = xr.DataArray(
        arr,
        coords=dict(frame=range(f), height=range(h), width=range(w)),
        dims=['frame', 'height', 'width'])
    varr.to_netcdf(os.path.dirname(filename) + os.sep + 'varr_mc_int.nc')
    return varr


def resave_varr(path, pattern='^varr_mc_int.tif$'):
    path = os.path.normpath(path)
    tiflist = []
    for dirpath, dirnames, fnames in os.walk(path):
        tifnames = filter(lambda fn: re.search(pattern, fn), fnames)
        tif_paths = [os.path.join(dirpath, tif) for tif in tifnames]
        tiflist += tif_paths
    for itif, tif_path in enumerate(tiflist):
        print("processing {:2d} of {:2d}".format(itif, len(tiflist)), end='\r')
        cur_var = tif_to_varray(tif_path)
        if not cur_var.sizes['height'] == 480 or not cur_var.sizes['width'] == 752:
            print("file {} has modified size: {}".format(
                tif_path, cur_var.sizes))


def plot_varr(varr):
    dvarr = hv.Dataset(varr, kdims=['width', 'height', 'frame'])
    layout = dvarr.to(hv.Image, ['width', 'height'])
    return layout


def save_cnmf(cnmf,
              dpath,
              save_pkl=True,
              from_pkl=False,
              unit_mask=None,
              meta_dict=None,
              order='C'):
    dpath = os.path.normpath(dpath)
    if from_pkl:
        with open(dpath + os.sep + 'cnm.pkl', 'rb') as f:
            cnmf = pkl.load(f)
    else:
        cnmf.dview = None
    if save_pkl:
        with open(dpath + os.sep + 'cnm.pkl', 'wb') as f:
            pkl.dump(cnmf, f)
    varr = xr.open_dataset(dpath + os.sep + 'varr_mc_int.nc')['varr_mc_int']
    f = varr.coords['frame']
    h = varr.coords['height']
    w = varr.coords['width']
    dims = cnmf.dims
    A = xr.DataArray(
        cnmf.A.toarray().reshape(dims + (-1, ), order=order),
        coords={'height': h,
                'width': w,
                'unit_id': range(cnmf.A.shape[-1])},
        dims=['height', 'width', 'unit_id'],
        name='A')
    C = xr.DataArray(
        cnmf.C,
        coords={'unit_id': range(cnmf.C.shape[0]),
                'frame': f},
        dims=['unit_id', 'frame'],
        name='C')
    S = xr.DataArray(
        cnmf.S,
        coords={'unit_id': range(cnmf.S.shape[0]),
                'frame': f},
        dims=['unit_id', 'frame'],
        name='S')
    YrA = xr.DataArray(
        cnmf.YrA,
        coords={'unit_id': range(cnmf.S.shape[0]),
                'frame': f},
        dims=['unit_id', 'frame'],
        name='YrA')
    b = xr.DataArray(
        cnmf.b.reshape(dims + (-1, ), order=order),
        coords={
            'height': h,
            'width': w,
            'background_id': range(cnmf.b.shape[-1])
        },
        dims=['height', 'width', 'background_id'],
        name='b')
    f = xr.DataArray(
        cnmf.f,
        coords={'background_id': range(cnmf.f.shape[0]),
                'frame': f},
        dims=['background_id', 'frame'],
        name='f')
    ds = xr.merge([A, C, S, YrA, b, f])
    if from_pkl:
        ds.to_netcdf(dpath + os.sep + 'cnm.nc', mode='a')
    else:
        if unit_mask is None:
            unit_mask = np.arange(ds.sizes['unit_id'])
        if meta_dict is not None:
            pathlist = os.path.normpath(dpath).split(os.sep)
            ds = ds.assign_coords(
                **dict([(cdname, pathlist[cdval])
                        for cdname, cdval in meta_dict.items()]))
        ds = ds.assign_attrs({
            'unit_mask': unit_mask,
            'file_path': dpath + os.sep + "cnm.nc"
        })
        ds.to_netcdf(dpath + os.sep + "cnm.nc")
    return ds


def save_varr(varr, dpath, name='varr_mc_int', meta_dict=None):
    dpath = os.path.normpath(dpath)
    ds = varr.to_dataset(name=name)
    if meta_dict is not None:
        pathlist = os.path.normpath(dpath).split(os.sep)
        ds = ds.assign_coords(**dict([(cdname, pathlist[cdval])
                                      for cdname, cdval in meta_dict.items()]))
    ds = ds.assign_attrs({'file_path': dpath + os.sep + name + '.nc'})
    ds.to_netcdf(dpath + os.sep + name + '.nc')
    return ds


def save_variable(var, fpath, fname, meta_dict=None):
    fpath = os.path.normpath(fpath)
    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.normpath(fpath).split(os.sep)
        ds = ds.assign_coords(**dict([(cdname, pathlist[cdval])
                                      for cdname, cdval in meta_dict.items()]))
    try:
        ds.to_netcdf(os.path.join(fpath, fname + '.nc'), mode='a')
    except FileNotFoundError:
        ds.to_netcdf(os.path.join(fpath, fname + '.nc'), mode='w')
    return ds


def update_meta(dpath, pattern=r'^varr_mc_int.nc$', meta_dict=None):
    for dirpath, dirnames, fnames in os.walk(dpath):
        fnames = filter(lambda fn: re.search(pattern, fn), fnames)
        for fname in fnames:
            f_path = os.path.join(dirpath, fname)
            pathlist = os.path.normpath(dirpath).split(os.sep)
            new_ds = xr.Dataset()
            with xr.open_dataset(f_path) as old_ds:
                new_ds.attrs = deepcopy(old_ds.attrs)
            new_ds = new_ds.assign_coords(
                **dict([(cdname, pathlist[cdval])
                        for cdname, cdval in meta_dict.items()]))
            new_ds = new_ds.assign_attrs(dict(file_path=f_path))
            new_ds.to_netcdf(f_path, mode='a')
            print("updated: {}".format(f_path))


# def resave_varr_again(dpath, pattern=r'^varr_mc_int.nc$'):
#     for dirpath, dirnames, fnames in os.walk(dpath):
#         fnames = filter(lambda fn: re.search(pattern, fn), fnames)
#         for fname in fnames:
#             f_path = os.path.join(dirpath, fname)
#             with xr.open_dataset(f_path) as old_ds:
#                 vname = list(old_ds.data_vars.keys())[0]
#                 if vname == 'varr_mc_int':
#                     continue
#                 print("resaving {}".format(f_path))
#                 ds = old_ds.load().copy()
#                 ds = ds.rename({vname: 'varr_mc_int'})
#             ds.to_netcdf(f_path, mode='w')


# def resave_cnmf(dpath, pattern=r'^cnm.nc$'):
#     for dirpath, fdpath, fpath in os.walk(dpath):
#         f_list = filter(lambda fn: re.search(pattern, fn), fpath)
#         for cnm_path in f_list:
#             cnm_path = os.path.join(dirpath, cnm_path)
#             cur_cnm = xr.open_dataset(cnm_path)
#             newds = xr.Dataset()
#             newds.assign_coords(session=cur_cnm.coords['session'])
#             newds.assign_coords(animal=cur_cnm.coords['animal'])
#             newds.assign_coords(session_id=cur_cnm.coords['session_id'])
#             fpath = str(cur_cnm.attrs['file_path'])
#             cur_cnm.close()
#             print("writing to ".format(fpath))
#             newds.to_netcdf(fpath, mode='a')


def save_movies(cnmf, dpath, Y=None, mask=None, Y_only=True, order='C'):
    try:
        cnmd = vars(cnmf)
    except TypeError:
        cnmd = cnmf
    dims = cnmd['dims']
    if not Y_only:
        print("calculating A * C")
        if mask is not None:
            A_dot_C = cnmd['A'].toarray()[:, mask].dot(
                cnmd['C'][mask, :]).astype(np.float32)
        else:
            A_dot_C = cnmd['A'].toarray().dot(cnmd['C']).astype(np.float32)
        print("calculating b * f")
        b_dot_f = cnmd['b'].dot(cnmd['f']).astype(np.float32)
        A_dot_C = xr.DataArray(
            A_dot_C.reshape(dims + (-1, ), order=order),
            coords={
                'height': range(dims[0]),
                'width': range(dims[1]),
                'frame': range(A_dot_C.shape[-1])
            },
            dims=['height', 'width', 'frame'],
            name='A_dot_C')
        b_dot_f = xr.DataArray(
            b_dot_f.reshape(dims + (-1, ), order=order),
            coords={
                'height': range(dims[0]),
                'width': range(dims[1]),
                'frame': range(b_dot_f.shape[-1])
            },
            dims=['height', 'width', 'frame'],
            name='b_dot_f')
    if Y is not None:
        Y = np.moveaxis(Y.astype(np.float32), 0, -1)
        if not isinstance(Y, xr.DataArray):
            Y = xr.DataArray(
                Y,
                coords={
                    'height': range(Y.shape[0]),
                    'width': range(Y.shape[1]),
                    'frame': range(Y.shape[2])
                },
                dims=['height', 'width', 'frame'],
                name='Y')
        if not Y_only:
            print("calculating Yres")
            Yres = Y.copy()
            Yres -= A_dot_C
            Yres -= b_dot_f
            Yres = Yres.rename('Yres')
    else:
        Yres = None
    if not Y_only:
        print("merging")
        ds = xr.merge([Y, A_dot_C, b_dot_f, Yres])
    else:
        ds = Y
    print("writing to disk")
    ds.to_netcdf(dpath + os.sep + "movies.nc")
    return ds


def save_cnmf_from_mat(matpath,
                       dpath,
                       vname="ms",
                       order='C',
                       dims=None,
                       T=None,
                       unit_mask=None,
                       meta_dict=None):
    dpath = os.path.normpath(dpath)
    mat = loadmat(matpath, squeeze_me=True, struct_as_record=False)
    try:
        cnmf = mat[vname]
    except KeyError:
        print("No variable with name {} was found in the .mat file: {}".format(
            vname, matpath))
        return
    if not dims:
        dims = (cnmf.options.d1, cnmf.options.d2)
        dims_coord = (list(range(dims[0])), list(range(dims[1])))
    else:
        dims_coord = (np.linspace(0, dims[0] - 1, cnmf.options.d1),
                      np.linspace(0, dims[1] - 1, cnmf.options.d2))
        dims = (cnmf.options.d1, cnmf.options.d2)
    if not T:
        T = cnmf.C.shape[1]
        T_coord = list(range(T))
    else:
        T_coord = np.linspace(0, T - 1, cnmf.C.shape[1])
        T = cnmf.C.shape[1]
    A = xr.DataArray(
        cnmf.A.reshape(dims + (-1, ), order=order),
        coords={
            'height': dims_coord[0],
            'width': dims_coord[1],
            'unit_id': range(cnmf.A.shape[-1])
        },
        dims=['height', 'width', 'unit_id'],
        name='A')
    C = xr.DataArray(
        cnmf.C,
        coords={'unit_id': range(cnmf.C.shape[0]),
                'frame': T_coord},
        dims=['unit_id', 'frame'],
        name='C')
    S = xr.DataArray(
        cnmf.S,
        coords={'unit_id': range(cnmf.S.shape[0]),
                'frame': T_coord},
        dims=['unit_id', 'frame'],
        name='S')
    if cnmf.b.any():
        b = xr.DataArray(
            cnmf.b.reshape(dims + (-1, ), order=order),
            coords={
                'height': dims_coord[0],
                'width': dims_coord[1],
                'background_id': range(cnmf.b.shape[-1])
            },
            dims=['height', 'width', 'background_id'],
            name='b')
    else:
        b = xr.DataArray(
            np.zeros(dims + (1, )),
            coords=dict(
                height=dims_coord[0], width=dims_coord[1], background_id=[0]),
            dims=['height', 'width', 'background_id'],
            name='b')
    if cnmf.f.any():
        f = xr.DataArray(
            cnmf.f,
            coords={'background_id': range(cnmf.f.shape[0]),
                    'frame': T_coord},
            dims=['background_id', 'frame'],
            name='f')
    else:
        f = xr.DataArray(
            np.zeros((1, T)),
            coords=dict(background_id=[0], frame=T_coord),
            dims=['background_id', 'frame'],
            name='f')
    ds = xr.merge([A, C, S, b, f])
    if unit_mask is None:
        unit_mask = np.arange(ds.sizes['unit_id'])
    if meta_dict is not None:
        pathlist = os.path.normpath(dpath).split(os.sep)
        ds = ds.assign_coords(
            **{cdname: pathlist[cdval]
               for cdname, cdval in meta_dict.items()})
    ds = ds.assign_attrs({
        'unit_mask': unit_mask,
        'file_path': dpath + os.sep + "cnm_from_mat.nc"
    })
    ds.to_netcdf(dpath + os.sep + "cnm_from_mat.nc")
    return ds


# def process_data(dpath, movpath, pltpath, roi):
#     params_movie = {
#         'niter_rig': 1,
#         'max_shifts': (20, 20),
#         'splits_rig': 28,
#         'num_splits_to_process_rig': None,
#         'strides': (48, 48),
#         'overlaps': (24, 24),
#         'splits_els': 28,
#         'num_splits_to_process_els': [14, None],
#         'upsample_factor_grid': 4,
#         'max_deviation_rigid': 3,
#         'p': 1,
#         'merge_thresh': 0.9,
#         'rf': 40,
#         'stride_cnmf': 20,
#         'K': 4,
#         'is_dendrites': False,
#         'init_method': 'greedy_roi',
#         'gSig': [10, 10],
#         'alpha_snmf': None,
#         'final_frate': 30
#     }
#     if not dpath.endswith(os.sep):
#         dpath = dpath + os.sep
#     if not os.path.isfile(dpath + 'mc.npz'):
#         # start parallel
#         c, dview, n_processes = cm.cluster.setup_cluster(
#             backend='local', n_processes=None, single_thread=False)
#         dpattern = 'msCam*.avi'
#         dlist = sorted(glob.glob(dpath + dpattern),
#                        key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
#         if not dlist:
#             print("No data found in the specified folder: " + dpath)
#             return
#         else:
#             vdlist = list()
#             for vname in dlist:
#                 vdlist.append(sio.vread(vname, as_grey=True))
#             mov_orig = cm.movie(
#                 np.squeeze(np.concatenate(vdlist, axis=0))).astype(np.float32)
#             # column correction
#             meanrow = np.mean(np.mean(mov_orig, 0), 0)
#             addframe = np.tile(meanrow, (mov_orig.shape[1], 1))
#             mov_cc = mov_orig - np.tile(addframe, (mov_orig.shape[0], 1, 1))
#             mov_cc = mov_cc - np.min(mov_cc)
#             # filter
#             mov_ft = mov_cc.copy()
#             for fid, fm in enumerate(mov_cc):
#                 mov_ft[fid] = ndi.uniform_filter(fm, 2) - ndi.uniform_filter(
#                     fm, 40)
#             mov_orig = (mov_orig - np.min(mov_orig)) / (
#                 np.max(mov_orig) - np.min(mov_orig))
#             mov_ft = (mov_ft - np.min(mov_ft)) / (
#                 np.max(mov_ft) - np.min(mov_ft))
#             np.save(dpath + 'mov_orig', mov_orig)
#             np.save(dpath + 'mov_ft', mov_ft)
#             del mov_orig, dlist, vdlist, mov_ft
#             mc_data = motion_correction.MotionCorrect(
#                 dpath + 'mov_ft.npy',
#                 0,
#                 dview=dview,
#                 max_shifts=params_movie['max_shifts'],
#                 niter_rig=params_movie['niter_rig'],
#                 splits_rig=params_movie['splits_rig'],
#                 num_splits_to_process_rig=params_movie[
#                     'num_splits_to_process_rig'],
#                 strides=params_movie['strides'],
#                 overlaps=params_movie['overlaps'],
#                 splits_els=params_movie['splits_els'],
#                 num_splits_to_process_els=params_movie[
#                     'num_splits_to_process_els'],
#                 upsample_factor_grid=params_movie['upsample_factor_grid'],
#                 max_deviation_rigid=params_movie['max_deviation_rigid'],
#                 shifts_opencv=True,
#                 nonneg_movie=False,
#                 roi=roi)
#             mc_data.motion_correct_rigid(save_movie=True)
#             mov_rig = cm.load(mc_data.fname_tot_rig)
#             np.save(dpath + 'mov_rig', mov_rig)
#             np.savez(
#                 dpath + 'mc',
#                 fname_tot_rig=mc_data.fname_tot_rig,
#                 templates_rig=mc_data.templates_rig,
#                 shifts_rig=mc_data.shifts_rig,
#                 total_templates_rig=mc_data.total_template_rig,
#                 max_shifts=mc_data.max_shifts,
#                 roi=mc_data.roi)
#             del mov_rig
#     else:
#         print("motion correction data already exist. proceed")
#     if not os.path.isfile(dpath + "cnm.npz"):
#         # start parallel
#         c, dview, n_processes = cm.cluster.setup_cluster(
#             backend='local', n_processes=None, single_thread=False)
#         fname_tot_rig = np.array_str(
#             np.load(dpath + 'mc.npz')['fname_tot_rig'])
#         mov, dims, T = cm.load_memmap(fname_tot_rig)
#         mov = np.reshape(mov.T, [T] + list(dims), order='F')
#         cnm = cnmf.CNMF(
#             n_processes,
#             k=params_movie['K'],
#             gSig=params_movie['gSig'],
#             merge_thresh=params_movie['merge_thresh'],
#             p=params_movie['p'],
#             dview=dview,
#             Ain=None,
#             rf=params_movie['rf'],
#             stride=params_movie['stride_cnmf'],
#             memory_fact=1,
#             method_init=params_movie['init_method'],
#             alpha_snmf=params_movie['alpha_snmf'],
#             only_init_patch=True,
#             gnb=1,
#             method_deconvolution='oasis')
#         cnm = cnm.fit(mov)
#         idx_comp, idx_comp_bad = estimate_components_quality(
#             cnm.C + cnm.YrA,
#             np.reshape(mov, dims + (T, ), order='F'),
#             cnm.A,
#             cnm.C,
#             cnm.b,
#             cnm.f,
#             params_movie['final_frate'],
#             Npeaks=10,
#             r_values_min=.7,
#             fitness_min=-40,
#             fitness_delta_min=-40)
#         A2 = cnm.A.tocsc()[:, idx_comp]
#         C2 = cnm.C[idx_comp]
#         cnm = cnmf.CNMF(
#             n_processes,
#             k=A2.shape,
#             gSig=params_movie['gSig'],
#             merge_thresh=params_movie['merge_thresh'],
#             p=params_movie['p'],
#             dview=dview,
#             Ain=A2,
#             Cin=C2,
#             f_in=cnm.f,
#             rf=None,
#             stride=None,
#             method_deconvolution='oasis')
#         cnm = cnm.fit(mov)
#         idx_comp, idx_comp_bad = estimate_components_quality(
#             cnm.C + cnm.YrA,
#             np.reshape(mov, dims + (T, ), order='F'),
#             cnm.A,
#             cnm.C,
#             cnm.b,
#             cnm.f,
#             params_movie['final_frate'],
#             Npeaks=10,
#             r_values_min=.75,
#             fitness_min=-50,
#             fitness_delta_min=-50)
#         cnm.A = cnm.A.tocsc()[:, idx_comp]
#         cnm.C = cnm.C[idx_comp]
#         cm.cluster.stop_server()
#         cnm.A = (cnm.A - np.min(cnm.A)) / (np.max(cnm.A) - np.min(cnm.A))
#         cnm.C = (cnm.C - np.min(cnm.C)) / (np.max(cnm.C) - np.min(cnm.C))
#         cnm.b = (cnm.b - np.min(cnm.b)) / (np.max(cnm.b) - np.min(cnm.b))
#         cnm.f = (cnm.f - np.min(cnm.f)) / (np.max(cnm.f) - np.min(cnm.f))
#         np.savez(
#             dpath + 'cnm',
#             A=cnm.A.todense(),
#             C=cnm.C,
#             b=cnm.b,
#             f=cnm.f,
#             YrA=cnm.YrA,
#             sn=cnm.sn,
#             dims=dims)
#     else:
#         print("cnm data already exist. proceed")
#     try:
#         A = cnm.A
#         C = cnm.C
#         dims = dims
#     except NameError:
#         A = np.load(dpath + 'cnm.npz')['A']
#         C = np.load(dpath + 'cnm.npz')['C']
#         dims = np.load(dpath + 'cnm.npz')['dims']
#     plot_components(A, C, dims, pltpath)

# def batch_process_data(animal_path, movroot, pltroot, roi):
#     for dirname, subdirs, files in os.walk(animal_path):
#         if files:
#             dirnamelist = dirname.split(os.sep)
#             dname = dirnamelist[-3] + '_' + dirnamelist[-2]
#             movpath = movroot + os.sep + dname
#             pltpath = pltroot + os.sep + dname
#             if not os.path.exists(movpath):
#                 os.mkdir(movpath)
#             if not os.path.exists(pltpath):
#                 os.mkdir(pltpath)
#             movpath = movpath + os.sep
#             pltpath = pltpath + os.sep
#             process_data(dirname, movpath, pltpath, roi)
#         else:
#             print("empty folder: " + dirname + " proceed")
