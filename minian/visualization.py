import functools as fct
import itertools as itt
import os
import subprocess
import xarray as xr

import colorcet as cc
import cv2
import dask
import dask.array as da
import ffmpeg
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import scipy.sparse as scisps
import sklearn.mixture
import skvideo.io
import time
from collections import OrderedDict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, Tuple, Callable, List
from uuid import uuid4
from bokeh.palettes import Category10_10, Viridis256
from dask.diagnostics import ProgressBar
from datashader import count_cat
from holoviews.operation.datashader import datashade, dynspread
from holoviews.streams import (
    BoxEdit,
    DoubleTap,
    Pipe,
    RangeXY,
    Selection1D,
    Stream,
    Tap,
)
from holoviews.util import Dynamic
from matplotlib import cm
from panel import widgets as pnwgt
from scipy import linalg
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial import cKDTree
from .cnmf import compute_AtC
from .motion_correction import apply_shifts
from .utilities import custom_arr_optimize, rechunk_like


class VArrayViewer:
    """
    Interactive visualization for movie data arrays.

    Hint
    ----
    .. figure:: img/vaviewer.png
        :width: 500px
        :align: left

    The visualization contains following panels from top to bottom:

    Play Toolbar
        A toolbar that controls playback of the video. Additionally, when the
        button "Update Mask" is clicked, the coordinates of the box drawn in
        *Current Frame* panel will be used to update the `mask` attribute of the
        `VArrayViewer` instance, which can be later used to subset the data. If
        multiple arrays are visualized and `layout` is `False`, then drop-down
        lists corresponding to each metadata dimensions will show up so the user
        can select which array to visualize.
    Current Frame
        Images of the current frame. If multiple movie array are passed in,
        multiple frames will be labeled and shown. To the side of each frame
        there is a histogram of intensity values. The "Box Select" tool can be
        used on the histogram to limit the range of intensity used for
        color-mapping. Additionally, the "Box Edit Tool" is available for use on
        the frame image, where you can hold "Shift" and draw a box, whose
        coordinates can be used to update the `mask` attribute of the
        `VarrayViewer` instance (remember to click "Update Mask" after drawing).
    Summary
        Summary statistics of each frame across time. Only shown if `summary` is
        not empty. The red vertical line indicate current frame.

    Attributes
    ----------
    mask : dict
        Instance attribute that can be retrieved and used to subset data later.
        Keys are `tuple` with values corresponding to each `meta_dims` and
        uniquely identify each input array. If `meta_dims` is empty then keys
        will be empty `tuple` as well. Values are `dict` mapping dimension names
        (of the arrays) to subsetting slices. The slices are in the plotting
        coorandinates and can be directly passed to `xr.DataArray.sel` method to
        subset data.
    """

    def __init__(
        self,
        varr: Union[xr.DataArray, List[xr.DataArray], xr.Dataset],
        framerate=30,
        summary=["mean"],
        meta_dims: List[str] = None,
        datashading=True,
        layout=False,
    ):
        """
        Parameters
        ----------
        varr : Union[xr.DataArray, List[xr.DataArray], xr.Dataset]
            Input array, list of arrays, or dataset to be visualized. Each array
            should contain dimensions "height", "width" and "frame". If a
            dataset, then the dimensions specified in `meta_dims` will be used
            as metadata dimensions that can uniquely identify each array. If a
            list, then a dimension "data_var" will be constructed and used as
            metadata dimension, and the `.name` attribute of each array will be
            used to identify each array.
        framerate : int, optional
            The framerate of playback when using the toolbar. By default `30`.
        summary : list, optional
            List of summary statistics to plot. The statistics should be one of
            `{"mean", "max", "min", "diff"}`. By default `["mean"]`.
        meta_dims : List[str], optional
            List of dimension names that can uniquely identify each input array
            in `varr`. Only used if `varr` is a `xr.Dataset`. By default `None`.
        datashading : bool, optional
            Whether to use datashading on the summary statistics. By default
            `True`.
        layout : bool, optional
            Whether to visualize all arrays together as layout. If `False` then
            only one array will be visualized and user can switch array using
            drop-down lists below the *Play Toolbar*. By default `False`.

        Raises
        ------
        NotImplementedError
            if `varr` is not a `xr.DataArray`, a `xr.Dataset` or a list of `xr.DataArray`
        """
        if isinstance(varr, list):
            for iv, v in enumerate(varr):
                varr[iv] = v.assign_coords(data_var=v.name)
            self.ds = xr.concat(varr, dim="data_var")
            meta_dims = ["data_var"]
        elif isinstance(varr, xr.DataArray):
            self.ds = varr.to_dataset()
        elif isinstance(varr, xr.Dataset):
            self.ds = varr
        else:
            raise NotImplementedError(
                "video array of type {} not supported".format(type(varr))
            )
        try:
            self.meta_dicts = OrderedDict(
                [(d, list(self.ds.coords[d].values)) for d in meta_dims]
            )
            self.cur_metas = OrderedDict(
                [(d, v[0]) for d, v in self.meta_dicts.items()]
            )
        except TypeError:
            self.meta_dicts = dict()
            self.cur_metas = dict()
        self._datashade = datashading
        self._layout = layout
        self.framerate = framerate
        self._f = self.ds.coords["frame"].values
        self._h = self.ds.sizes["height"]
        self._w = self.ds.sizes["width"]
        self.mask = dict()
        CStream = Stream.define(
            "CStream",
            f=param.Integer(
                default=int(self._f.min()), bounds=(self._f.min(), self._f.max())
            ),
        )
        self.strm_f = CStream()
        self.str_box = BoxEdit()
        self.widgets = self._widgets()
        if type(summary) is list:
            summ_all = {
                "mean": self.ds.mean(["height", "width"]),
                "max": self.ds.max(["height", "width"]),
                "min": self.ds.min(["height", "width"]),
                "diff": self.ds.diff("frame").mean(["height", "width"]),
            }
            try:
                summ = {k: summ_all[k] for k in summary}
            except KeyError:
                print("{} Not understood for specifying summary".format(summary))
            if summ:
                print("computing summary")
                sum_list = []
                for k, v in summ.items():
                    sum_list.append(v.compute().assign_coords(sum_var=k))
                summary = xr.concat(sum_list, dim="sum_var")
        self.summary = summary
        if layout:
            self.ds_sub = self.ds
            self.sum_sub = self.summary
        else:
            self.ds_sub = self.ds.sel(**self.cur_metas)
            try:
                self.sum_sub = self.summary.sel(**self.cur_metas)
            except AttributeError:
                self.sum_sub = self.summary
        self.pnplot = pn.panel(self.get_hvobj())

    def get_hvobj(self):
        def get_im_ovly(meta):
            def img(f, ds):
                return hv.Image(ds.sel(frame=f).compute(), kdims=["width", "height"])

            try:
                curds = self.ds_sub.sel(**meta).rename("_".join(meta.values()))
            except ValueError:
                curds = self.ds_sub
            fim = fct.partial(img, ds=curds)
            im = hv.DynamicMap(fim, streams=[self.strm_f]).opts(
                frame_width=500, aspect=self._w / self._h, cmap="Viridis"
            )
            self.xyrange = RangeXY(source=im).rename(x_range="w", y_range="h")
            if not self._layout:
                hv_box = hv.Polygons([]).opts(
                    style={"fill_alpha": 0.3, "line_color": "white"}
                )
                self.str_box = BoxEdit(source=hv_box)
                im_ovly = im * hv_box
            else:
                im_ovly = im

            def hist(f, w, h, ds):
                if w and h:
                    cur_im = hv.Image(
                        ds.sel(frame=f).compute(), kdims=["width", "height"]
                    ).select(height=h, width=w)
                else:
                    cur_im = hv.Image(
                        ds.sel(frame=f).compute(), kdims=["width", "height"]
                    )
                return hv.operation.histogram(cur_im, num_bins=50).opts(
                    xlabel="fluorescence", ylabel="freq"
                )

            fhist = fct.partial(hist, ds=curds)
            his = hv.DynamicMap(fhist, streams=[self.strm_f, self.xyrange]).opts(
                frame_height=int(500 * self._h / self._w), width=150, cmap="Viridis"
            )
            im_ovly = (im_ovly << his).map(lambda p: p.opts(style=dict(cmap="Viridis")))
            return im_ovly

        if self._layout and self.meta_dicts:
            im_dict = OrderedDict()
            for meta in itt.product(*list(self.meta_dicts.values())):
                mdict = {k: v for k, v in zip(list(self.meta_dicts.keys()), meta)}
                im_dict[meta] = get_im_ovly(mdict)
            ims = hv.NdLayout(im_dict, kdims=list(self.meta_dicts.keys()))
        else:
            ims = get_im_ovly(self.cur_metas)
        if self.summary is not None:
            hvsum = (
                hv.Dataset(self.sum_sub)
                .to(hv.Curve, kdims=["frame"])
                .overlay("sum_var")
            )
            if self._datashade:
                hvsum = datashade_ndcurve(hvsum, kdim="sum_var")
            try:
                hvsum = hvsum.layout(list(self.meta_dicts.keys()))
            except:
                pass
            vl = hv.DynamicMap(lambda f: hv.VLine(f), streams=[self.strm_f]).opts(
                style=dict(color="red")
            )
            summ = (hvsum * vl).map(
                lambda p: p.opts(frame_width=500, aspect=3), [hv.RGB, hv.Curve]
            )
            hvobj = (ims + summ).cols(1)
        else:
            hvobj = ims
        return hvobj

    def show(self) -> pn.layout.Column:
        """
        Return visualizations that can be directly displayed.

        Returns
        -------
        pn.layout.Column
            Resulting visualizations containing both plots and toolbars.
        """
        return pn.layout.Column(self.widgets, self.pnplot)

    def _widgets(self):
        w_play = pnwgt.Player(
            length=len(self._f), interval=10, value=0, width=650, height=90
        )

        def play(f):
            if not f.old == f.new:
                self.strm_f.event(f=int(self._f[f.new]))

        w_play.param.watch(play, "value")
        w_box = pnwgt.Button(
            name="Update Mask", button_type="primary", width=100, height=30
        )
        w_box.param.watch(self._update_box, "clicks")
        if not self._layout:
            wgt_meta = {
                d: pnwgt.Select(name=d, options=v, height=45, width=120)
                for d, v in self.meta_dicts.items()
            }

            def make_update_func(meta_name):
                def _update(x):
                    self.cur_metas[meta_name] = x.new
                    self._update_subs()

                return _update

            for d, wgt in wgt_meta.items():
                cur_update = make_update_func(d)
                wgt.param.watch(cur_update, "value")
            wgts = pn.layout.WidgetBox(w_box, w_play, *list(wgt_meta.values()))
        else:
            wgts = pn.layout.WidgetBox(w_box, w_play)
        return wgts

    def _update_subs(self):
        self.ds_sub = self.ds.sel(**self.cur_metas)
        if self.sum_sub is not None:
            self.sum_sub = self.summary.sel(**self.cur_metas)
        self.pnplot.objects[0].object = self.get_hvobj()

    def _update_box(self, click):
        box = self.str_box.data
        self.mask.update(
            {
                tuple(self.cur_metas.values()): {
                    "height": slice(box["y0"][0], box["y1"][0]),
                    "width": slice(box["x0"][0], box["x1"][0]),
                }
            }
        )


class CNMFViewer:
    """
    Interactive visualization for CNMF results.

    Hint
    ----
    .. figure:: img/cnmfviewer.png
        :width: 1000px

    The visualization can be divided into two parts vertically:

    Spatial
        Top part of the visualization. Shows spatial plots at a given time. From
        left to right:

        Spatial Footprints
            Shows the spatial footprints of all cells. The "Box Select" tool can
            be used in this panel to select a subset of cells to visualize for
            both the *Isolated Activities* panel and the *Temporal Activities*
            panel.
        Isolated Activities
            Shows activities of selected cells only. If the "UseAC" checkbox
            under *General Toolbox* is enabled, then the `AtC` variable computed
            with the selected cells will be visualized at the given frame (See
            :func:`minian.cnmf.compute_AtC`). Otherwise the spatial footprints
            of the cells will be plotted, which would be invariant across time.
            The "unit_id" coordinates for each cell are shown on top of each
            cell.
        Original Movie
            Shows a single frame of an arbitrary movie data supplied in `org`.

    Temporal
        Bottom part of the visualization. Shows temporal activities across time
        and various toolboxes. From left to right:

        General Toolbox
            Contains the following tools:

            * "Refresh" button, will refresh all visualization when clicked.
            * "Load Data" button, will load all data in memory for faster
              visualization, can be very memory-demanding.
            * "UseAC" checkbox, whether to plot spatial-temporal activities for
              the *Isolated Activities* panel.
            * "ShowC", "ShowS", "Normalize" checkboxes, whether to show the
              calcium traces, the spike signals, or to normalize both traces
              to unit range for each cell.
            * "Group" dropbox, "Previous Group" and "Next Group" buttons, select
              the group of cells to visualize. The grouping is controled by
              `sortNN` parameter.
            * Playback toolbar, used to control which timepoint is visualized.
            * Additional metadata dropdown, if the input dataset contains
              additional metadata dimensions then dropdown will show up so
              user can select which dataset to visualize.
        Temporal Activities
            Shows temporal activities of selected subset of cells. The red
            vertical line indicate current frame. Additionally user can
            double-click anywhere in the plot to move current frame to that
            location.
        Manual Label
            Shows tools to carry out manual labeling of cells. User can either
            manually assign unit label using the dropdown for each cell, or
            select some cells with the checkboxes corresponding to the
            "unit_id", and then merge or discard the units using the buttons.
            The "Unit Label" dropdowns should update and refelect the merging or
            discarding actions.

    Attributes
    ----------
    unit_labels : xr.DataArray
        1d array whose values represent the result of manual refinement of
        cells. The "unit_id" coordinate of this array is identical to input
        data. The values of this array can be interpreted as new "unit_id" after
        the manual refinement, where duplicated values indicate merged cells,
        and values of -1 indicate discarded cells.
    """

    def __init__(
        self,
        minian: Optional[xr.Dataset] = None,
        A: Optional[xr.DataArray] = None,
        C: Optional[xr.DataArray] = None,
        S: Optional[xr.DataArray] = None,
        org: Optional[xr.DataArray] = None,
        sortNN=True,
    ):
        """
        Parameters
        ----------
        minian : xr.Dataset, optional
            Input minian dataset containing all necessary variables. If `None`
            then all other arguments should be supplied. By default `None`.
        A : xr.DataArray, optional
            Spatial footprints of cells. If `None` then it will be retrieved as
            `minian["A"]`. By default `None`.
        C : xr.DataArray, optional
            Calcium dynamic of cells. If `None` then it will be retrieved as
            `minian["C"]`. By default `None`.
        S : xr.DataArray, optional
            Deconvolved spikes of cells. If `None` then it will be retrieved as
            `minian["S"]`. By default `None`.
        org : xr.DataArray, optional
            Arbitrary movie data to be visualized along with results of CNMF. If
            `None` then it will be retrieved as `minian["org"]`. If this array
            contains dimensions other than "height", "width" or "frame" then
            they will be used as metadata dimensions. By default `None`.
        sortNN : bool, optional
            Whether to sort the units using :func:`NNsort` so that cells close
            together will appear in same group for visualization. If `False`
            then cells are simply grouped in 5 by ascending "unit_id". By
            default `True`.
        """
        self._A = A if A is not None else minian["A"]
        self._C = C if C is not None else minian["C"]
        self._S = S if S is not None else minian["S"]
        self._org = org if org is not None else minian["org"]
        try:
            self.unit_labels = minian["unit_labels"].compute()
        except:
            self.unit_labels = xr.DataArray(
                self._A["unit_id"].values.copy(),
                dims=self._A["unit_id"].dims,
                coords=self._A["unit_id"].coords,
            ).rename("unit_labels")
        self._C_norm = xr.apply_ufunc(
            normalize,
            self._C.chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self._C.dtype],
        )
        self._S_norm = xr.apply_ufunc(
            normalize,
            self._S.chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self._C.dtype],
        )
        self.cents = centroid(self._A, verbose=True)
        print("computing sum projection")
        with ProgressBar():
            self.Asum = self._A.sum("unit_id").compute()
        self._NNsort = sortNN
        self._normalize = False
        self._useAC = True
        self._showC = True
        self._showS = True
        meta_dims = list(set(self._org.dims) - {"frame", "height", "width"})
        self.meta_dicts = {d: list(self._org.coords[d].values) for d in meta_dims}
        self.metas = {d: v[0] for d, v in self.meta_dicts.items()}
        if self._NNsort:
            try:
                self.cents["NNord"] = self.cents.groupby(
                    meta_dims, group_keys=False
                ).apply(NNsort)
            except ValueError:
                self.cents["NNord"] = NNsort(self.cents)
            NNcoords = self.cents.set_index(meta_dims + ["unit_id"])[
                "NNord"
            ].to_xarray()
            self._A = self._A.assign_coords(NNord=NNcoords)
            self._C = self._C.assign_coords(NNord=NNcoords)
            self._S = self._S.assign_coords(NNord=NNcoords)
            self._C_norm = self._C_norm.assign_coords(NNord=NNcoords)
            self._S_norm = self._S_norm.assign_coords(NNord=NNcoords)
        self.update_subs()
        self.strm_f = DoubleTap(rename=dict(x="f"))
        self.strm_f.add_subscriber(self.callback_f)
        self.strm_uid = Selection1D()
        self.strm_uid.add_subscriber(self.callback_uid)
        Stream_usub = Stream.define("Stream_usub", usub=param.List())
        self.strm_usub = Stream_usub()
        self.strm_usub.add_subscriber(self.callback_usub)
        self.usub_sel = self.strm_usub.usub
        self._AC = self._org.sel(**self.metas)
        self._mov = self._org.sel(**self.metas)
        self.pipAC = Pipe([])
        self.pipmov = Pipe([])
        self.pipusub = Pipe([])
        self.wgt_meta = self._meta_wgt()
        self.wgt_spatial_all = self._spatial_all_wgt()
        self.spatial_all = self._spatial_all()
        self.temp_comp_sub = self._temp_comp_sub(self._u[:5])
        self.wgt_man = self._man_wgt()
        self.wgt_temp_comp = self._temp_comp_wgt()

    def update_subs(self):
        self.A_sub = self._A.sel(**self.metas)
        self.C_sub = self._C.sel(**self.metas)
        self.S_sub = self._S.sel(**self.metas)
        self.org_sub = self._org.sel(**self.metas)
        self.C_norm_sub = self._C_norm.sel(**self.metas)
        self.S_norm_sub = self._S_norm.sel(**self.metas)
        if self._NNsort:
            self.A_sub = self.A_sub.sortby("NNord")
            self.C_sub = self.C_sub.sortby("NNord")
            self.S_sub = self.S_sub.sortby("NNord")
            self.C_norm_sub = self.C_norm_sub.sortby("NNord")
            self.S_norm_sub = self.S_norm_sub.sortby("NNord")
        self._h = (
            self.A_sub.isel(unit_id=0)
            .dropna("height", how="all")
            .coords["height"]
            .values
        )
        self._w = (
            self.A_sub.isel(unit_id=0).dropna("width", how="all").coords["width"].values
        )
        self._f = self.C_sub.isel(unit_id=0).dropna("frame").coords["frame"].values
        self._u = self.C_sub.isel(frame=0).dropna("unit_id").coords["unit_id"].values
        if self.meta_dicts:
            sub = pd.concat(
                [self.cents[d] == v for d, v in self.metas.items()], axis="columns"
            ).all(axis="columns")
            self.cents_sub = self.cents[sub]
        else:
            self.cents_sub = self.cents

    def compute_subs(self, clicks=None):
        self.A_sub = self.A_sub.compute()
        self.C_sub = self.C_sub.compute()
        self.S_sub = self.S_sub.compute()
        self.org_sub = self.org_sub.compute()
        self.C_norm_sub = self.C_norm_sub.compute()
        self.S_norm_sub = self.S_norm_sub.compute()

    def update_all(self, clicks=None):
        self.update_subs()
        self.strm_uid.event(index=[])
        self.strm_f.event(x=0)
        self.update_spatial_all()

    def callback_uid(self, index=None):
        self.update_temp()
        self.update_AC()
        self.update_usub_lab()

    def callback_f(self, f, y):
        if len(self._AC) > 0 and len(self._mov) > 0:
            fidx = np.abs(self._f - f).argmin()
            f = self._f[fidx]
            if self._useAC:
                AC = self._AC.sel(frame=f)
            else:
                AC = self._AC
            mov = self._mov.sel(frame=f)
            self.pipAC.send(AC)
            self.pipmov.send(mov)
            try:
                self.wgt_temp_comp[1].value = int(fidx)
            except AttributeError:
                pass
        else:
            self.pipAC.send([])
            self.pipmov.send([])

    def callback_usub(self, usub=None):
        self.update_temp_comp_sub(usub)
        self.update_AC(usub)
        self.update_usub_lab(usub)

    def _meta_wgt(self):
        wgt_meta = {
            d: pnwgt.Select(name=d, options=v, height=45, width=120)
            for d, v in self.meta_dicts.items()
        }

        def make_update_func(meta_name):
            def _update(x):
                self.metas[meta_name] = x.new
                self.update_subs()

            return _update

        for d, wgt in wgt_meta.items():
            cur_update = make_update_func(d)
            wgt.param.watch(cur_update, "value")
        wgt_update = pnwgt.Button(
            name="Refresh", button_type="primary", height=30, width=120
        )
        wgt_update.param.watch(self.update_all, "clicks")
        wgt_load = pnwgt.Button(
            name="Load Data", button_type="danger", height=30, width=120
        )
        wgt_load.param.watch(self.compute_subs, "clicks")
        return pn.layout.WidgetBox(
            *(list(wgt_meta.values()) + [wgt_update, wgt_load]), width=150
        )

    def show(self) -> pn.layout.Column:
        """
        Return visualizations that can be directly displayed.

        Returns
        -------
        pn.layout.Column
            Resulting visualizations containing both plots and toolboxes.
        """
        return pn.layout.Column(
            self.spatial_all,
            pn.layout.Row(
                pn.layout.Column(
                    pn.layout.Row(self.wgt_meta, self.wgt_spatial_all),
                    self.wgt_temp_comp,
                ),
                self.temp_comp_sub,
                self.wgt_man,
            ),
        )

    def _temp_comp_sub(self, usub=None):
        if usub is None:
            usub = self.strm_usub.usub
        if self._normalize:
            C, S = self.C_norm_sub, self.S_norm_sub
        else:
            C, S = self.C_sub, self.S_sub
        cur_temp = dict()
        if self._showC:
            cur_temp["C"] = hv.Dataset(
                C.sel(unit_id=usub)
                .compute()
                .rename("Intensity (A. U.)")
                .dropna("frame", how="all")
            ).to(hv.Curve, "frame")
        if self._showS:
            cur_temp["S"] = hv.Dataset(
                S.sel(unit_id=usub)
                .compute()
                .rename("Intensity (A. U.)")
                .dropna("frame", how="all")
            ).to(hv.Curve, "frame")
        cur_vl = hv.DynamicMap(
            lambda f, y: hv.VLine(f) if f else hv.VLine(0), streams=[self.strm_f]
        ).opts(style=dict(color="red"))
        cur_cv = hv.Curve([], kdims=["frame"], vdims=["Internsity (A.U.)"])
        self.strm_f.source = cur_cv
        h_cv = len(self._w) // 8
        w_cv = len(self._w) * 2
        temp_comp = (
            cur_cv
            * datashade_ndcurve(
                hv.HoloMap(cur_temp, "trace")
                .collate()
                .overlay("trace")
                .grid("unit_id")
                .add_dimension("time", 0, 0),
                "trace",
            )
            .opts(plot=dict(shared_xaxis=True))
            .map(
                lambda p: p.opts(plot=dict(frame_height=h_cv, frame_width=w_cv)), hv.RGB
            )
            * cur_vl
        )
        temp_comp[temp_comp.keys()[0]] = temp_comp[temp_comp.keys()[0]].opts(
            plot=dict(height=h_cv + 75)
        )
        return pn.panel(temp_comp)

    def update_temp_comp_sub(self, usub=None):
        self.temp_comp_sub.object = self._temp_comp_sub(usub).object
        self.wgt_man.objects = self._man_wgt().objects

    def update_norm(self, norm):
        self._normalize = norm.new
        self.update_temp_comp_sub()

    def _temp_comp_wgt(self):
        if self.strm_uid.index:
            cur_idxs = self.strm_uid.index
        else:
            cur_idxs = self._u
        ntabs = np.ceil(len(cur_idxs) / 5)
        sub_idxs = np.array_split(cur_idxs, ntabs)
        idxs_dict = OrderedDict(
            [("group{}".format(i), g.tolist()) for i, g in enumerate(sub_idxs)]
        )
        def_idxs = list(idxs_dict.values())[0]
        wgt_grp = pnwgt.Select(
            name="", options=idxs_dict, width=120, height=30, value=def_idxs
        )

        def update_usub(usub):
            self.usub_sel = []
            self.strm_usub.event(usub=usub.new)

        wgt_grp.param.watch(update_usub, "value")
        wgt_grp.value = def_idxs
        self.strm_usub.event(usub=def_idxs)
        wgt_grp_prv = pnwgt.Button(
            name="Previous Group", width=120, height=30, button_type="primary"
        )

        def prv(clicks):
            cur_val = wgt_grp.value
            ig = list(idxs_dict.values()).index(cur_val)
            try:
                prv_val = idxs_dict[list(idxs_dict.keys())[ig - 1]]
                wgt_grp.value = prv_val
            except:
                pass

        wgt_grp_prv.param.watch(prv, "clicks")
        wgt_grp_nxt = pnwgt.Button(
            name="Next Group", width=120, height=30, button_type="primary"
        )

        def nxt(clicks):
            cur_val = wgt_grp.value
            ig = list(idxs_dict.values()).index(cur_val)
            try:
                nxt_val = idxs_dict[list(idxs_dict.keys())[ig + 1]]
                wgt_grp.value = nxt_val
            except:
                pass

        wgt_grp_nxt.param.watch(nxt, "clicks")
        wgt_norm = pnwgt.Checkbox(
            name="Normalize", value=self._normalize, width=120, height=10
        )
        wgt_norm.param.watch(self.update_norm, "value")
        wgt_showC = pnwgt.Checkbox(
            name="ShowC", value=self._showC, width=120, height=10
        )

        def callback_showC(val):
            self._showC = val.new
            self.update_temp_comp_sub()

        wgt_showC.param.watch(callback_showC, "value")
        wgt_showS = pnwgt.Checkbox(
            name="ShowS", value=self._showS, width=120, height=10
        )

        def callback_showS(val):
            self._showS = val.new
            self.update_temp_comp_sub()

        wgt_showS.param.watch(callback_showS, "value")
        wgt_play = pnwgt.Player(length=len(self._f), interval=10, value=0, width=280)

        def play(f):
            if not f.old == f.new:
                self.strm_f.event(x=self._f[f.new])

        wgt_play.param.watch(play, "value")
        wgt_groups = pn.layout.Row(
            pn.layout.WidgetBox(wgt_norm, wgt_showC, wgt_showS, wgt_grp, width=150),
            pn.layout.WidgetBox(wgt_grp_prv, wgt_grp_nxt, width=150),
        )
        return pn.layout.Column(wgt_groups, wgt_play)

    def _man_wgt(self):
        usub = self.strm_usub.usub
        usub.sort()
        usub.reverse()
        ulabs = self.unit_labels.sel(unit_id=usub).values
        wgt_sel = {
            uid: pnwgt.Select(
                name="Unit Label",
                options=usub + [-1] + ulabs.tolist(),
                value=ulb,
                height=50,
                width=80,
            )
            for uid, ulb in zip(usub, ulabs)
        }

        def callback_ulab(value, uid):
            self.unit_labels.loc[uid] = value.new

        for uid, sel in wgt_sel.items():
            cb = fct.partial(callback_ulab, uid=uid)
            sel.param.watch(cb, "value")
        wgt_check = {
            uid: pnwgt.Checkbox(
                name="Unit ID: {}".format(uid), value=False, height=50, width=100
            )
            for uid in usub
        }

        def callback_chk(val, uid):
            if not val.old == val.new:
                if val.new:
                    self.usub_sel.append(uid)
                else:
                    self.usub_sel.remove(uid)

        for uid, chk in wgt_check.items():
            cb = fct.partial(callback_chk, uid=uid)
            chk.param.watch(cb, "value")
        wgt_discard = pnwgt.Button(
            name="Discard Selected", button_type="primary", width=180
        )

        def callback_discard(clicks):
            for uid in self.usub_sel:
                wgt_sel[uid].value = -1

        wgt_discard.param.watch(callback_discard, "clicks")
        wgt_merge = pnwgt.Button(
            name="Merge Selected", button_type="primary", width=180
        )

        def callback_merge(clicks):
            for uid in self.usub_sel:
                wgt_sel[uid].value = self.usub_sel[0]

        wgt_merge.param.watch(callback_merge, "clicks")
        return pn.layout.Column(
            pn.layout.WidgetBox(wgt_discard, wgt_merge, width=200),
            pn.layout.Row(
                pn.layout.WidgetBox(*wgt_check.values(), width=100),
                pn.layout.WidgetBox(*wgt_sel.values(), width=100),
            ),
        )

    def update_temp_comp_wgt(self):
        self.wgt_temp_comp.objects = self._temp_comp_wgt().objects

    def update_temp(self):
        self.update_temp_comp_wgt()

    def update_AC(self, usub=None):
        if usub is None:
            usub = self.strm_usub.usub
        if usub:
            if self._useAC:
                umask = (self.A_sub.sel(unit_id=usub) > 0).any("unit_id")
                A_sub = self.A_sub.sel(unit_id=usub).where(umask, drop=True).fillna(0)
                C_sub = self.C_sub.sel(unit_id=usub)
                AC = xr.apply_ufunc(
                    da.dot,
                    A_sub,
                    C_sub,
                    input_core_dims=[
                        ["height", "width", "unit_id"],
                        ["unit_id", "frame"],
                    ],
                    output_core_dims=[["height", "width", "frame"]],
                    dask="allowed",
                )
                self._AC = AC.compute()
                wndh, wndw = AC.coords["height"].values, AC.coords["width"].values
                window = self.A_sub.sel(
                    height=slice(wndh.min(), wndh.max()),
                    width=slice(wndw.min(), wndw.max()),
                )
                self._AC = self._AC.reindex_like(window).fillna(0)
                self._mov = (self.org_sub.reindex_like(window)).compute()
            else:
                self._AC = self.A_sub.sel(unit_id=usub).sum("unit_id")
                self._mov = self.org_sub
            self.strm_f.event(x=0)
        else:
            self._AC = xr.DataArray([])
            self._mov = xr.DataArray([])
            self.strm_f.event(x=0)

    def update_usub_lab(self, usub=None):
        if usub is None:
            usub = self.strm_usub.usub
        if usub:
            self.pipusub.send(self.cents_sub[self.cents_sub["unit_id"].isin(usub)])
        else:
            self.pipusub.send([])

    def _spatial_all_wgt(self):
        wgt_useAC = pnwgt.Checkbox(
            name="UseAC", value=self._useAC, width=120, height=15
        )

        def callback_useAC(val):
            self._useAC = val.new
            self.update_AC()

        wgt_useAC.param.watch(callback_useAC, "value")
        return pn.layout.WidgetBox(wgt_useAC, width=150)

    def _spatial_all(self):
        metas = self.metas
        Asum = hv.Image(self.Asum.sel(**metas), ["width", "height"]).opts(
            plot=dict(frame_height=len(self._h), frame_width=len(self._w)),
            style=dict(cmap="Viridis"),
        )
        cents = (
            hv.Dataset(
                self.cents_sub.drop(list(self.meta_dicts.keys()), axis="columns"),
                kdims=["width", "height", "unit_id"],
            )
            .to(hv.Points, ["width", "height"])
            .opts(
                style=dict(
                    alpha=0.1,
                    line_alpha=0,
                    size=5,
                    nonselection_alpha=0.1,
                    selection_alpha=0.9,
                )
            )
            .collate()
            .overlay("unit_id")
            .opts(plot=dict(tools=["hover", "box_select"]))
        )
        self.strm_uid.source = cents
        fim = fct.partial(hv.Image, kdims=["width", "height"])
        AC = hv.DynamicMap(fim, streams=[self.pipAC]).opts(
            plot=dict(frame_height=len(self._h), frame_width=len(self._w)),
            style=dict(cmap="Viridis"),
        )
        mov = hv.DynamicMap(fim, streams=[self.pipmov]).opts(
            plot=dict(frame_height=len(self._h), frame_width=len(self._w)),
            style=dict(cmap="Viridis"),
        )
        lab = fct.partial(hv.Labels, kdims=["width", "height"], vdims=["unit_id"])
        ulab = hv.DynamicMap(lab, streams=[self.pipusub]).opts(
            style=dict(text_color="red")
        )
        return pn.panel(Asum * cents + AC * ulab + mov)

    def update_spatial_all(self):
        self.spatial_all.objects = self._spatial_all().objects


class AlignViewer:
    """
    Interactive visualization of cross-registration resuls.

    Hint
    ----
    .. image:: img/alignviewer.png
        :width: 700px

    This class visualize the result of cross-registration by color-mapping
    spatial footprints of cells from three selected sessions as red, green and
    blue channel and show an overlay image. In addition to the overlay image,
    following tools are available:

    Channel Selector
        Contains "sessionR", "sessionG", and "sessionB" dropdowns, allowing the
        user to select which sessions are colormapped to each channel.
    Display Settings
        Contains the following tools:

        * "erode" dropdown, set window size of an optional erode operation
          applied to the spatial footprints for display to reduce overlaps.
        * "show matched" and "show unmatched" checkboxes, set whether to show
          cells that are matched or not matched across all three selected sessions.
    Metadata Selector
        If additional metadata are present, dropdowns corresponding to each
        metadata dimensions will be shown.

    """

    def __init__(
        self,
        minian_ds: xr.Dataset,
        cents: pd.DataFrame,
        mappings: pd.DataFrame,
        shiftds: xr.Dataset,
        brt_offset=0,
    ) -> None:
        """
        Parameters
        ----------
        minian_ds : xr.Dataset
            Input dataset. Should contain `minian_ds["A"]`.
        cents : pd.DataFrame
            Input centroids of cells.
        mappings : pd.DataFrame
            Input mappings of cells.
        shiftds : xr.Dataset
            Input dataset of shift results. Should contain `shiftds["shifts"]`.
        brt_offset : int, optional
            Brightness offset added on top of the color-mapped image. Useful to
            make the image visually brighter. By default `0`.
        """
        # init
        self.minian_ds = minian_ds
        self.cents = cents
        self.mappings = mappings
        self.shiftds = shiftds
        self.brt_offset = brt_offset
        A = self.minian_ds["A"]
        self.shifts = rechunk_like(self.shiftds["shifts"], A)
        self.Ash = apply_shifts(A, self.shifts, fill=0)
        # option widgets
        self.erode = 3
        wgt_er = pnwgt.Select(name="erode", options=np.arange(0, 20).tolist(), value=3)
        wgt_er.param.watch(self.cb_update_erd, "value")
        self.show_ma = True
        wgt_ma = pnwgt.Checkbox(name="show matched", value=True)
        wgt_ma.param.watch(self.cb_showma, "value")
        self.show_uma = True
        wgt_uma = pnwgt.Checkbox(name="show unmatched", value=True)
        wgt_uma.param.watch(self.cb_showuma, "value")
        self.wgt_opt = pn.layout.WidgetBox(wgt_er, wgt_ma, wgt_uma)
        self.processA()
        # handling meta
        try:
            self.meta_dict = {
                col: c.unique().tolist() for col, c in mappings["meta"].iteritems()
            }
        except KeyError:
            self.meta_dict = None
        if self.meta_dict:
            self.meta = {d: v[0] for d, v in self.meta_dict.items()}
            wgt_meta = [
                pnwgt.Select(name=dim, options=vals)
                for dim, vals in self.meta_dict.items()
            ]
            for w in wgt_meta:
                w.param.watch(lambda v, n=w.name: self.cb_update_meta(n, v), "value")
            self.wgt_meta = pn.layout.WidgetBox(*wgt_meta)
        else:
            self.wgt_meta = None
        self.update_meta()
        # sessionRGB
        sess = list(mappings["session"].columns)
        self.sess_rgb = {"r": sess[0], "g": sess[0], "b": sess[0]}
        wgt_sess = {
            c: pnwgt.Select(name="session{}".format(c.upper()), options=sess)
            for c in ["r", "g", "b"]
        }
        for wname, w in wgt_sess.items():
            w.param.watch(lambda v, n=wname: self.cb_update_rgb(n, v), "value")
        self.wgt_rgb = pn.layout.WidgetBox(*list(wgt_sess.values()))
        self.plot = self.update_plot()

    def processA(self):
        A = self.Ash
        if self.erode >= 3:
            A = xr.apply_ufunc(
                cv2.erode,
                A,
                input_core_dims=[["height", "width"]],
                output_core_dims=[["height", "width"]],
                vectorize=True,
                dask="parallelized",
                kwargs={"kernel": np.ones((self.erode, self.erode))},
                output_dtypes=[float],
            )
        self.dataA = xr.apply_ufunc(
            norm,
            A,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    def update_plot(self):
        Adict = {
            c: self.curA.sel(session=self.sess_rgb[c])
            .dropna("unit_id", how="all")
            .compute()
            for c in self.sess_rgb.keys()
        }
        map_sub = self.curmap["session"][list(self.sess_rgb.values())].dropna(how="all")
        map_sub = map_sub.loc[:, ~map_sub.columns.duplicated()]
        ma_mask = map_sub.notnull().all(axis="columns")
        imdict = {
            c: np.zeros((A.sizes["height"], A.sizes["width"])) for c, A in Adict.items()
        }
        if self.show_ma:
            ma_map = map_sub.loc[ma_mask]
            for c, im in imdict.items():
                uids = ma_map[self.sess_rgb[c]].values
                imdict[c] = im + Adict[c].sel(unit_id=uids).sum("unit_id").compute()
        if self.show_uma:
            uma_map = map_sub.loc[~ma_mask]
            for c, im in imdict.items():
                uids = uma_map[self.sess_rgb[c]].dropna().values
                imdict[c] = im + Adict[c].sel(unit_id=uids).sum("unit_id").compute()
        cmaps = {
            "r": cc.m_linear_kryw_0_100_c71,
            "g": cc.m_linear_green_5_95_c69,
            "b": cc.m_linear_blue_5_95_c73,
        }
        for c, im in imdict.items():
            imdict[c] = cm.ScalarMappable(cmap=cmaps[c]).to_rgba(im)
        im_ovly = xr.DataArray(
            np.clip(imdict["r"] + imdict["g"] + imdict["b"] + self.brt_offset, 0, 1),
            dims=["height", "width", "rgb"],
            coords={
                "height": self.curA.coords["height"].values,
                "width": self.curA.coords["width"].values,
            },
        )
        im_opts = {
            "frame_height": self.curA.sizes["height"],
            "frame_width": self.curA.sizes["width"],
        }
        return pn.panel(
            hv.RGB(
                (
                    im_ovly.coords["width"],
                    im_ovly.coords["height"],
                    im_ovly[:, :, 0],
                    im_ovly[:, :, 1],
                    im_ovly[:, :, 2],
                    im_ovly[:, :, 3],
                ),
                kdims=["width", "height"],
            ).opts(**im_opts)
        )

    def update_meta(self):
        if self.meta_dict:
            self.curA = self.dataA.sel(**self.meta).persist()
            self.curmap = (
                self.mappings.set_index([("meta", d) for d in self.meta.keys()])
                .loc[tuple(self.meta.values())]
                .reset_index()
            )
        else:
            self.curA = self.dataA.persist()
            self.curmap = self.mappings

    def cb_update_erd(self, val):
        self.erode = val.new
        self.processA()
        self.update_meta()
        self.plot.object = self.update_plot().object

    def cb_update_meta(self, dim, val):
        self.meta[dim] = val.new
        self.update_meta()
        self.plot.object = self.update_plot().object

    def cb_update_rgb(self, ch, ss):
        self.sess_rgb[ch] = ss.new
        self.plot.object = self.update_plot().object

    def cb_showma(self, val):
        self.show_ma = val.new
        self.plot.object = self.update_plot().object

    def cb_showuma(self, val):
        self.show_uma = val.new
        self.plot.object = self.update_plot().object

    def show(self) -> pn.layout.Row:
        """
        Return visualizations that can be directly displayed.

        Returns
        -------
        pn.layout.Row
            Resulting visualizations containing both plots and toolbars.
        """
        return pn.layout.Row(
            self.plot, pn.layout.Column(self.wgt_meta, self.wgt_rgb, self.wgt_opt)
        )


def write_vid_blk(arr, vpath, options):
    uid = uuid4()
    vname = "{}.mp4".format(uid)
    fpath = os.path.join(vpath, vname)
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=0)
    writer = skvideo.io.FFmpegWriter(
        fpath, outputdict={"-" + k: v for k, v in options.items()}
    )
    for fm in arr:
        writer.writeFrame(fm)
    writer.close()
    return fpath


def write_video(
    arr: xr.DataArray,
    vname: Optional[str] = None,
    vpath: Optional[str] = ".",
    norm=True,
    options={"crf": "18", "preset": "ultrafast"},
) -> str:
    """
    Write a video from a movie array using `python-ffmpeg`.

    Parameters
    ----------
    arr : xr.DataArray
        Input movie array. Should have dimensions: ("frame", "height", "width")
        and should only be chunked along the "frame" dimension.
    vname : str, optional
        The name of output video. If `None` then a random one will be generated
        using :func:`uuid4.uuid`. By default `None`.
    vpath : str, optional
        The path to the folder containing the video. By default `"."`.
    norm : bool, optional
        Whether to normalize the values of the input array such that they span
        the full pixel depth range (0, 255). By default `True`.
    options : dict, optional
        Optional output arguments passed to `ffmpeg`. By default `{"crf": "18",
        "preset": "ultrafast"}`.

    Returns
    -------
    fname : str
        The absolute path to the video file.

    See Also
    --------
    ffmpeg.output
    """
    if not vname:
        vname = "{}.mp4".format(uuid4())
    fname = os.path.join(vpath, vname)
    if norm:
        arr_opt = fct.partial(
            custom_arr_optimize, rename_dict={"rechunk": "merge_restricted"}
        )
        with dask.config.set(array_optimize=arr_opt):
            arr = arr.astype(np.float32)
            arr_max = arr.max().compute().values
            arr_min = arr.min().compute().values
        den = arr_max - arr_min
        arr -= arr_min
        arr /= den
        arr *= 255
    arr = arr.clip(0, 255).astype(np.uint8)
    w, h = arr.sizes["width"], arr.sizes["height"]
    process = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="gray", s="{}x{}".format(w, h))
        .filter("pad", int(np.ceil(w / 2) * 2), int(np.ceil(h / 2) * 2))
        .output(fname, pix_fmt="yuv420p", vcodec="libx264", r=30, **options)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for blk in arr.data.blocks:
        process.stdin.write(np.array(blk).tobytes())
    process.stdin.close()
    process.wait()
    return fname


def concat_video_recursive(vlist, vname=None):
    if not len(vlist) > 1:
        return vlist[0]
    if len(vlist) > 256:
        vlist = np.array_split(vlist, 256)
        vlist = [concat_video_recursive(list(v)) for v in vlist]
    vpath = os.path.dirname(vlist[0])
    streams = [ffmpeg.input(p) for p in vlist]
    if vname is None:
        vname = "{}.mp4".format(uuid4())
    fpath = os.path.join(vpath, vname)
    ffmpeg.concat(*streams).output(fpath).run(overwrite_output=True)
    for vp in vlist:
        os.remove(vp)
    return fpath


def generate_videos(
    varr: xr.DataArray,
    Y: xr.DataArray,
    A: Optional[xr.DataArray] = None,
    C: Optional[xr.DataArray] = None,
    AC: Optional[xr.DataArray] = None,
    nfm_norm: int = None,
    gain=1.5,
    vpath=".",
    vname="minian.mp4",
    options={"crf": "18", "preset": "ultrafast"},
) -> str:
    """
    Generate a video visualizaing the result of minian pipeline.

    The resulting video contains four parts: Top left is a original reference
    movie supplied as `varr`; Top right is the input to CNMF algorithm supplied
    as `Y`; Bottom right is a movie `AC` representing cellular activities as
    computed by :func:`minian.cnmf.compute_AtC`; Bottom left is a residule movie
    computed as the difference between `Y` and `AC`. Since the CNMF algorithm
    contains various arbitrary scaling process, a normalizing scalar is computed
    with least square using a subset of frames from `Y` and `AC` such that their
    numerical values matches.

    Parameters
    ----------
    varr : xr.DataArray
        Input reference movie data. Should have dimensions ("frame", "height",
        "width"), and should only be chunked along "frame" dimension.
    Y : xr.DataArray
        Movie data representing input to CNMF algorithm. Should have dimensions
        ("frame", "height", "width"), and should only be chunked along "frame"
        dimension.
    A : xr.DataArray, optional
        Spatial footprints of cells. Only used if `AC` is `None`. By default
        `None`.
    C : xr.DataArray, optional
        Temporal activities of cells. Only used if `AC` is `None`. By default
        `None`.
    AC : xr.DataArray, optional
        Spatial-temporal activities of cells. Should have dimensions ("frame",
        "height", "width"), and should only be chunked along "frame" dimension.
        If `None` then both `A` and `C` should be supplied and
        :func:`minian.cnmf.compute_AtC` will be used to compute this variable.
        By default `None`.
    nfm_norm : int, optional
        Number of frames to randomly draw from `Y` and `AC` to compute the
        normalizing factor with least square. By default `None`.
    gain : float, optional
        A gain factor multiplied to `Y`. Useful to make the results visually
        brighter. By default `1.5`.
    vpath : str, optional
        Desired folder containing the resulting video. By default `"."`.
    vname : str, optional
        Desired name of the video. By default `"minian.mp4"`.
    options : dict, optional
        Output options for `ffmpeg`, passed directly to :func:`write_video`. By
        default `{"crf": "18", "preset": "ultrafast"}`.

    Returns
    -------
    fname : str
        Absolute path of the resulting video.
    """
    if AC is None:
        print("generating traces")
        AC = compute_AtC(A, C)
    print("normalizing")
    gain = 255 / Y.max().compute().values * gain
    Y = Y * gain
    if nfm_norm is not None:
        norm_idx = np.sort(
            np.random.choice(np.arange(Y.sizes["frame"]), size=nfm_norm, replace=False)
        )
        Y_sub = Y.isel(frame=norm_idx).values.reshape(-1)
        AC_sub = scisps.csc_matrix(AC.isel(frame=norm_idx).values.reshape((-1, 1)))
        lsqr = scisps.linalg.lsqr(AC_sub, Y_sub)
        norm_factor = lsqr[0].item()
        del Y_sub, AC_sub
    else:
        norm_factor = gain
    AC = AC * norm_factor
    res = Y - AC
    print("writing videos")
    vid = xr.concat(
        [
            xr.concat([varr, Y], "width", coords="minimal"),
            xr.concat([res, AC], "width", coords="minimal"),
        ],
        "height",
        coords="minimal",
    )
    return write_video(vid, vname, vpath, norm=False, options=options)


def datashade_ndcurve(
    ovly: hv.NdOverlay, kdim: Optional[Union[str, List[str]]] = None, spread=False
) -> hv.Overlay:
    """
    Apply datashading to an overlay of curves with legends.

    Parameters
    ----------
    ovly : hv.NdOverlay
        The input overlay of curves.
    kdim : Union[str, List[str]], optional
        Key dimensions of the overlay. If `None` then the first key dimension of
        `ovly` will be used. By default `None`.
    spread : bool, optional
        Whether to apply :func:`holoviews.operation.datashader.dynspread` to the
        result. By default `False`.

    Returns
    -------
    hvres : hv.Overlay
        Resulting overlay of datashaded curves and points (for legends).
    """
    if not kdim:
        kdim = ovly.kdims[0].name
    var = np.unique(ovly.dimension_values(kdim)).tolist()
    color_key = [(v, Category10_10[iv]) for iv, v in enumerate(var)]
    color_pts = hv.NdOverlay(
        {
            k: hv.Points([0, 0], label=str(k)).opts(style=dict(color=v))
            for k, v in color_key
        }
    )
    ds_ovly = datashade(
        ovly,
        aggregator=count_cat(kdim),
        color_key=dict(color_key),
        min_alpha=200,
        normalization="linear",
    )
    if spread:
        ds_ovly = dynspread(ds_ovly)
    return ds_ovly * color_pts


def construct_G(g: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Construct a convolving matrix from AR coefficients.

    Parameters
    ----------
    g : np.ndarray
        Input AR coefficients.
    T : np.ndarray
        Number of time samples of the AR process.

    Returns
    -------
    G : np.ndarray
        A `T` x `T` matrix that can be used to multiply with a timeseries to
        convolve the AR process.

    See Also
    --------
    minian.cnmf.update_temporal :
        for more background on the role of AR process in the pipeline
    """
    cur_c, cur_r = np.zeros(T), np.zeros(T)
    cur_c[0] = 1
    cur_r[0] = 1
    cur_c[1 : len(g) + 1] = -g
    return linalg.toeplitz(cur_c, cur_r)


def normalize(a: np.ndarray) -> np.ndarray:
    """
    Normalize an input array to range (0, 1) using :func:`numpy.interp`.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    a_norm : np.ndarray
        Normalized array.
    """
    return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, +1))


def norm(a: np.ndarray) -> np.ndarray:
    """
    Normalize an input array to range (0, 1) avoiding division-by-zero.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    a_norm : np.ndarray
        Normalized array. If there is only one unique value in `a` then it is
        returned unchanged.
    """
    amax = np.nanmax(a)
    amin = np.nanmin(a)
    diff = amax - amin
    if diff > 0:
        return (a - amin) / (amax - amin)
    else:
        return a


def convolve_G(s: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Convolve an AR process to input timeseries.

    Despite the name, only AR coefficients are needed as input. The convolving
    matrix will be computed using :func:`construct_G`.

    Parameters
    ----------
    s : np.ndarray
        The input timeseries, presumably representing spike signals.
    g : np.ndarray
        The AR coefficients.

    Returns
    -------
    c : np.ndarray
        Convolved timeseries, presumably representing calcium dynamics.

    See Also
    --------
    minian.cnmf.update_temporal :
        for more background on the role of AR process in the pipeline
    """
    G = construct_G(g, len(s))
    try:
        c = np.linalg.inv(G).dot(s)
    except np.linalg.LinAlgError:
        c = s.copy()
    return c


def construct_pulse_response(
    g: np.ndarray, length=500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a model pulse response corresponding to certain AR coefficients.

    Parameters
    ----------
    g : np.ndarray
        The AR coefficients.
    length : int, optional
        Number of timepoints in output. By default `500`.

    Returns
    -------
    s : np.ndarray
        Model spike with shape `(length,)`, zero everywhere except the first
        timepoint.
    c : np.ndarray
        Model convolved calcium response, with same shape as `s`.

    See Also
    --------
    minian.cnmf.update_temporal :
        for more background on the role of AR process in the pipeline
    """
    s = np.zeros(length)
    s[np.arange(0, length, 500)] = 1
    c = convolve_G(s, g)
    return s, c


def centroid(A: xr.DataArray, verbose=False) -> pd.DataFrame:
    """
    Compute centroids of spatial footprint of each cell.

    Parameters
    ----------
    A : xr.DataArray
        Input spatial footprints.
    verbose : bool, optional
        Whether to print message and progress bar. By default `False`.

    Returns
    -------
    cents_df : pd.DataFrame
        Centroid of spatial footprints for each cell. Has columns "unit_id",
        "height", "width" and any other additional metadata dimension.
    """

    def rel_cent(im):
        im_nan = np.isnan(im)
        if im_nan.all():
            return np.array([np.nan, np.nan])
        if im_nan.any():
            im = np.nan_to_num(im)
        cent = np.array(center_of_mass(im))
        return cent / im.shape

    gu_rel_cent = da.gufunc(
        rel_cent,
        signature="(h,w)->(d)",
        output_dtypes=float,
        output_sizes=dict(d=2),
        vectorize=True,
    )
    cents = xr.apply_ufunc(
        gu_rel_cent,
        A.chunk(dict(height=-1, width=-1)),
        input_core_dims=[["height", "width"]],
        output_core_dims=[["dim"]],
        dask="allowed",
    ).assign_coords(dim=["height", "width"])
    if verbose:
        print("computing centroids")
        with ProgressBar():
            cents = cents.compute()
    cents_df = (
        cents.rename("cents")
        .to_series()
        .dropna()
        .unstack("dim")
        .rename_axis(None, axis="columns")
        .reset_index()
    )
    h_rg = (A.coords["height"].min().values, A.coords["height"].max().values)
    w_rg = (A.coords["width"].min().values, A.coords["width"].max().values)
    cents_df["height"] = cents_df["height"] * (h_rg[1] - h_rg[0]) + h_rg[0]
    cents_df["width"] = cents_df["width"] * (w_rg[1] - w_rg[0]) + w_rg[0]
    return cents_df


def visualize_preprocess(
    fm: xr.DataArray, fn: Optional[Callable] = None, include_org=True, **kwargs
) -> hv.HoloMap:
    """
    Generalized visualization of preprocessing functions.

    This function facilitates parameter exploration of preprocessing functions
    by plotting a single frame before and after the application of the function,
    along with a contour plot. All keyword arguments not listed below are passed
    directly to `fn`.

    Parameters
    ----------
    fm : xr.DataArray
        The input frame.
    fn : Callable, optional
        The function to apply. If `None` then the original frame are visualized
        unchanged. By default `None`.
    include_org : bool, optional
        Whether to include the original frame in the visualization. By default
        `True`.

    Returns
    -------
    hvres : hv.HoloMap
        The resulting visualization containing images and contour plots.

    See Also
    --------
    minian.preprocessing
    """
    fh, fw = fm.sizes["height"], fm.sizes["width"]
    asp = fw / fh
    opts_im = {
        "plot": {
            "frame_width": 500,
            "aspect": asp,
            "title": "Image {label} {group} {dimensions}",
        },
        "style": {"cmap": "viridis"},
    }
    opts_cnt = {
        "plot": {
            "frame_width": 500,
            "aspect": asp,
            "title": "Contours {label} {group} {dimensions}",
        },
        "style": {"cmap": "viridis"},
    }

    def _vis(f):
        im = hv.Image(f, kdims=["width", "height"]).opts(**opts_im)
        cnt = hv.operation.contours(im).opts(**opts_cnt)
        return im, cnt

    if fn is not None:
        pkey = kwargs.keys()
        pval = kwargs.values()
        im_dict = dict()
        cnt_dict = dict()
        for params in itt.product(*pval):
            fm_res = fn(fm, **dict(zip(pkey, params)))
            cur_im, cur_cnt = _vis(fm_res)
            cur_im = cur_im.relabel("After")
            cur_cnt = cur_cnt.relabel("After")
            p_str = tuple(
                [str(p) if not isinstance(p, (int, float)) else p for p in params]
            )
            im_dict[p_str] = cur_im
            cnt_dict[p_str] = cur_cnt
        hv_im = Dynamic(hv.HoloMap(im_dict, kdims=list(pkey)).opts(**opts_im))
        hv_cnt = datashade(
            hv.HoloMap(cnt_dict, kdims=list(pkey)), precompute=True, cmap=Viridis256
        ).opts(**opts_cnt)
        if include_org:
            im, cnt = _vis(fm)
            im = im.relabel("Before").opts(**opts_im)
            cnt = (
                datashade(cnt, precompute=True, cmap=Viridis256)
                .relabel("Before")
                .opts(**opts_cnt)
            )
        return (im + cnt + hv_im + hv_cnt).cols(2)
    else:
        im, cnt = _vis(fm)
        im = im.relabel("Before")
        cnt = cnt.relabel("Before")
        return im + cnt


def visualize_seeds(
    max_proj: xr.DataArray, seeds: pd.DataFrame, mask: Optional[str] = None
) -> hv.Overlay:
    """
    Visualization of seeds.

    This function plot seeds on top of a max projection. It can also visualize
    certain refining step of seeds by coloring the filtered-out seeds in red.

    Parameters
    ----------
    max_proj : xr.DataArray
        Max projection used as the background of the plot.
    seeds : pd.DataFrame
        The seed dataframe.
    mask : str, optional
        The name of the mask of seeds to visualize. If specified, then `seeds`
        must contain a boolean column with the same name. By default `None`.

    Returns
    -------
    hvres : hv.Overlay
        The resuling overlay of seeds and max projection.

    See Also
    --------
    minian.initialization
    """
    h, w = max_proj.sizes["height"], max_proj.sizes["width"]
    asp = w / h
    pt_cmap = {True: "white", False: "red"}
    opts_im = dict(plot=dict(frame_width=600, aspect=asp), style=dict(cmap="Viridis"))
    opts_pts = dict(
        plot=dict(
            frame_width=600,
            aspect=asp,
            size_index="seeds",
            color_index=mask,
            tools=["hover"],
        ),
        style=dict(fill_alpha=0.8, line_alpha=0, cmap=pt_cmap),
    )
    if mask:
        vdims = ["seeds", mask]
    else:
        vdims = ["seeds"]
        opts_pts["style"]["color"] = "white"
    im = hv.Image(max_proj, kdims=["width", "height"])
    pts = hv.Points(seeds, kdims=["width", "height"], vdims=vdims)
    return im.opts(**opts_im) * pts.opts(**opts_pts)


def visualize_gmm_fit(
    values: np.ndarray, gmm: sklearn.mixture.GaussianMixture, bins: int
) -> hv.Overlay:
    """
    Visualization of the Gaussian mixture model fit.

    This function visualize GMM fit by plotting the fitted gaussian curves on
    top of the histograms of values.

    Parameters
    ----------
    values : np.ndarray
        The raw values to which GMM is fitted.
    gmm : sklearn.mixture.GaussianMixture
        The fitted GMM model object.
    bins : int
        Number of bins when plotting the histogram.

    Returns
    -------
    hvres : hv.Overlay
        The resulting visualization.

    See Also
    --------
    minian.initialization.gmm_refine
    """

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

    hist = np.histogram(values, bins=bins, density=True)
    gss_dict = dict()
    for igss, (mu, sig) in enumerate(zip(gmm.means_, gmm.covariances_)):
        gss = gaussian(hist[1], np.asscalar(mu), np.asscalar(np.sqrt(sig)))
        gss_dict[igss] = hv.Curve((hist[1], gss))
    return (
        hv.Histogram(((hist[0] - hist[0].min()) / np.ptp(hist[0]), hist[1])).opts(
            style=dict(alpha=0.6, fill_color="gray")
        )
        * hv.NdOverlay(gss_dict)
    ).opts(plot=dict(height=350, width=500))


def visualize_spatial_update(
    A_dict: dict,
    C_dict: dict,
    kdims: Optional[Union[str, List[str]]] = None,
    norm=True,
    datashading=True,
) -> hv.HoloMap:
    """
    Visualization of spatial update.

    This function facilitates parameter exploration for spatial update by
    plotting the resulting spatial footprints and binarized spatial footprints
    from different runs of spatial update for a subset of cells, along with
    their corresponding temporal activities.

    Parameters
    ----------
    A_dict : dict
        A dictionary containing resulting spatial footprints from different runs
        of spatial update. Keys should be tuple containing the values of
        parameters that uniquely identify each run. Values should be spatial
        footprints of type `xr.DataArray`.
    C_dict : dict
        A dictionary containing temporal activities of each cells in the same
        format as `A_dict`. The temporal activities of cells are not expected to
        change across different runs of spatial update, except the number of
        cells may be different due to dropping of cells in the update process.
    kdims : Union[str, List[str]], optional
        Names of key dimensions identifying the parameter space. Should have
        same length as the keys in `A_dict` and `C_dict`. If `None` then a
        dimension names "dummy" will be created and the visualization can be
        used to visualize restults across cells. By default `None`.
    norm : bool, optional
        Whether to normalize the temporal activities of each cell to range (0,
        1) for visualization. By default `True`.
    datashading : bool, optional
        Whether to apply datashading to temporal activities of cells. By default
        `True`.

    Returns
    -------
    hvres : hv.HoloMap
        Resulting visualization.

    See Also
    --------
    minian.cnmf.update_spatial
    """
    if not kdims:
        A_dict = dict(dummy=A_dict)
        C_dict = dict(dummy=C_dict)
    hv_pts_dict, hv_A_dict, hv_Ab_dict, hv_C_dict = (dict(), dict(), dict(), dict())
    for key, A in A_dict.items():
        A = A.compute()
        C = C_dict[key]
        if norm:
            C = xr.apply_ufunc(
                normalize,
                C.chunk(dict(frame=-1)),
                input_core_dims=[["frame"]],
                output_core_dims=[["frame"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[C.dtype],
            )
        C = C.compute()
        h, w = A.sizes["height"], A.sizes["width"]
        cents_df = centroid(A)
        hv_pts_dict[key] = hv.Points(
            cents_df, kdims=["width", "height"], vdims=["unit_id"]
        ).opts(
            plot=dict(tools=["hover"]), style=dict(fill_alpha=0.2, line_alpha=0, size=8)
        )
        hv_A_dict[key] = hv.Image(
            A.sum("unit_id").rename("A"), kdims=["width", "height"]
        )
        hv_Ab_dict[key] = hv.Image(
            (A > 0).sum("unit_id").rename("A_bin"), kdims=["width", "height"]
        )
        hv_C_dict[key] = hv.Dataset(C.rename("C")).to(hv.Curve, kdims="frame")
    hv_pts = Dynamic(hv.HoloMap(hv_pts_dict, kdims=kdims))
    hv_A = Dynamic(hv.HoloMap(hv_A_dict, kdims=kdims))
    hv_Ab = Dynamic(hv.HoloMap(hv_Ab_dict, kdims=kdims))
    hv_C = (
        hv.HoloMap(hv_C_dict, kdims=kdims)
        .collate()
        .grid("unit_id")
        .add_dimension("time", 0, 0)
    )
    if datashading:
        hv_C = datashade(hv_C)
    else:
        hv_C = Dynamic(hv_C)
    hv_A = hv_A.opts(frame_width=400, aspect=w / h, colorbar=True, cmap="viridis")
    hv_Ab = hv_Ab.opts(frame_width=400, aspect=w / h, colorbar=True, cmap="viridis")
    hv_C = hv_C.map(
        lambda cr: cr.opts(frame_width=500, frame_height=50),
        hv.RGB if datashading else hv.Curve,
    )
    return (
        hv.NdLayout(
            {"pseudo-color": (hv_pts * hv_A), "binary": (hv_pts * hv_Ab)},
            kdims="Spatial Matrix",
        ).cols(1)
        + hv_C.relabel("Temporal Components")
    )


def visualize_temporal_update(
    YA_dict: dict,
    C_dict: dict,
    S_dict: dict,
    g_dict: dict,
    sig_dict: dict,
    A_dict: dict,
    kdims: Optional[Union[str, List[str]]] = None,
    norm=True,
    datashading=True,
) -> hv.HoloMap:
    """
    Visualization of temporal update.

    This function facilitates parameter exploration for temporal update by
    plotting various temporal traces along with a model calcium response and the
    spatial footprint for each cell across different runs of temporal update.
    Four traces are plotted: "Raw Signal" correspond to the `YrA` variable,
    "Fitted Calcium Trace" correspond to `C` after update, "Fitted Spikes"
    correspond to `S` after update, and "Fitted Signal" correspond to `C + b0 +
    c0` after update. See :func:`minian.cnmf.update_temporal` for interpretation
    of each variable.

    Parameters
    ----------
    YA_dict : dict
        A dictionary containing the `YrA` variables in the same format as
        `C_dict`. The `YrA` variable is not updated and is not expected to be
        different across different runs of temporal update.
    C_dict : dict
        A dictionary containing resulting calcium traces (`C_new`) from
        different runs of temporal update. Keys should be tuple containing the
        values of parameters that uniquely identify each run. Values should be
        temporal traces of type `xr.DataArray`.
    S_dict : dict
        A dictionary containing resulting deconvolved spike traces (`S_new`)
        from different runs of temporal update, in the same format as `C_dict`.
    g_dict : dict
        A dictionary containing resulting AR coefficients (`g`) from different
        runs of temporal update, in the same format as `C_dict`.
    sig_dict : dict
        A dictionary containing resulting fitted signals (`C_new + b0_new +
        c0_new`) from different runs of temporal update, in the same format as
        `C_dict`.
    A_dict : dict
        A dictionary containing spatial footprint of cells in the same format as
        `C_dict`. The spatial footprints of cells are note expected to change
        across different runs of temporal update, except the number of cells may
        be different due to dropping of cells in the update process.
    kdims : Union[str, List[str]], optional
        Names of key dimensions identifying the parameter space. Should have
        same length as the keys in `C_dict` etc. If `None` then a dimension
        names "dummy" will be created and the visualization can be used to
        visualize restults across cells. By default `None`.
    norm : bool, optional
        Whether to normalize the temporal activities of each cell to range (0,
        1) for visualization. By default `True`.
    datashading : bool, optional
        Whether to apply datashading to temporal activities of cells. By default
        `True`.

    Returns
    -------
    hvres : hv.HoloMap
        Resulting visualization.

    See Also
    --------
    minian.cnmf.update_temporal
    """
    inputs = [YA_dict, C_dict, S_dict, sig_dict, g_dict]
    if not kdims:
        inputs = [dict(dummy=i) for i in inputs]
        A_dict = dict(dummy=A_dict)
    input_dict = {k: [i[k] for i in inputs] for k in inputs[0].keys()}
    hv_YA, hv_C, hv_S, hv_sig, hv_C_pul, hv_S_pul, hv_A = [dict() for _ in range(7)]
    for k, ins in input_dict.items():
        if norm:
            ins[:-1] = [
                xr.apply_ufunc(
                    normalize,
                    i.chunk(dict(frame=-1)),
                    input_core_dims=[["frame"]],
                    output_core_dims=[["frame"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[i.dtype],
                )
                for i in ins[:-1]
            ]
        ins[:] = [i.compute() for i in ins]
        ya, c, s, sig, g = ins
        f_crd = ya.coords["frame"]
        pul_crd = f_crd.values[:500]
        s_pul, c_pul = xr.apply_ufunc(
            construct_pulse_response,
            g,
            input_core_dims=[["lag"]],
            output_core_dims=[["t"], ["t"]],
            vectorize=True,
            kwargs=dict(length=len(pul_crd)),
            output_sizes=dict(t=len(pul_crd)),
        )
        s_pul, c_pul = (s_pul.assign_coords(t=pul_crd), c_pul.assign_coords(t=pul_crd))
        if norm:
            c_pul = xr.apply_ufunc(
                normalize,
                c_pul.chunk(dict(t=-1)),
                input_core_dims=[["t"]],
                output_core_dims=[["t"]],
                dask="parallelized",
                output_dtypes=[c_pul.dtype],
            ).compute()
        pul_range = (
            f_crd.min(),
            int(np.around(f_crd.min() + (f_crd.max() - f_crd.min()) / 2)),
        )
        hv_S_pul[k], hv_C_pul[k] = [
            (hv.Dataset(tr.rename("Response (A.U.)")).to(hv.Curve, kdims=["t"]))
            for tr in [s_pul, c_pul]
        ]
        hv_YA[k] = hv.Dataset(ya.rename("Intensity (A.U.)")).to(
            hv.Curve, kdims=["frame"]
        )
        if c.sizes["unit_id"] > 0:
            hv_C[k], hv_S[k], hv_sig[k] = [
                (
                    hv.Dataset(tr.rename("Intensity (A.U.)")).to(
                        hv.Curve, kdims=["frame"]
                    )
                )
                for tr in [c, s, sig]
            ]
        hv_A[k] = hv.Dataset(A_dict[k].rename("A")).to(
            hv.Image, kdims=["width", "height"]
        )
        h, w = A_dict[k].sizes["height"], A_dict[k].sizes["width"]
    hvobjs = [hv_YA, hv_C, hv_S, hv_sig, hv_C_pul, hv_S_pul, hv_A]
    hvobjs[:] = [hv.HoloMap(hvobj, kdims=kdims).collate() for hvobj in hvobjs]
    hv_unit = {
        "Raw Signal": hvobjs[0],
        "Fitted Calcium Trace": hvobjs[1],
        "Fitted Spikes": hvobjs[2],
        "Fitted Signal": hvobjs[3],
    }
    hv_pul = {"Simulated Calcium": hvobjs[4], "Simulated Spike": hvobjs[5]}
    hv_unit = hv.HoloMap(hv_unit, kdims="traces").collate().overlay("traces")
    hv_pul = hv.HoloMap(hv_pul, kdims="traces").collate().overlay("traces")
    hv_A = Dynamic(hvobjs[6])
    if datashading:
        hv_unit = datashade_ndcurve(hv_unit, "traces")
    else:
        hv_unit = Dynamic(hv_unit)
    hv_pul = Dynamic(hv_pul)
    hv_unit = hv_unit.map(
        lambda p: p.opts(plot=dict(frame_height=400, frame_width=1000))
    )
    hv_pul = hv_pul.opts(plot=dict(frame_width=500, aspect=w / h)).redim(
        t=hv.Dimension("t", soft_range=pul_range)
    )
    hv_A = hv_A.opts(
        plot=dict(frame_width=500, aspect=w / h), style=dict(cmap="Viridis")
    )
    return (
        hv_unit.relabel("Current Unit: Temporal Traces")
        + hv.NdLayout(
            {"Simulated Pulse Response": hv_pul, "Spatial Footprint": hv_A},
            kdims="Current Unit",
        )
    ).cols(1)


def NNsort(cents: pd.DataFrame) -> pd.Series:
    """
    Sort centroids of cells into close-by groups.

    Walk through centroids of cells using a nearest neighbors tree such that the
    resulting walk order can be used to sort cells into close-by groups.

    Parameters
    ----------
    cents : pd.DataFrame
        Input centroids of cells. Should contain column "height" and "width".

    Returns
    -------
    result : pd.Series
        A series with same index as input `cents` whose values represent the
        order of nearest-neighbor walk.
    """
    cents_hw = cents[["height", "width"]]
    kdtree = cKDTree(cents_hw)
    idu_start = cents_hw.sum(axis="columns").idxmin()
    result = pd.Series(0, index=cents.index)
    remain_list = cents.index.tolist()
    idu_next = idu_start
    NNord = 0
    while remain_list:
        result.loc[idu_next] = NNord
        remain_list.remove(idu_next)
        for k in range(1, int(np.ceil(np.log2(len(result)))) + 1):
            qry = kdtree.query(cents_hw.loc[idu_next], 2 ** k)
            NNs = qry[1][np.isfinite(qry[0])].squeeze()
            NNs = NNs[np.sort(np.unique(NNs, return_index=True)[1])]
            NNs = np.array(result.iloc[NNs].index)
            NN_idxs = np.argwhere(np.isin(NNs, remain_list, assume_unique=True))
            if len(NN_idxs) > 0:
                NN = NNs[NN_idxs[0]][0]
                idu_next = NN
                NNord = NNord + 1
                break
    return result


def visualize_motion(motion: xr.DataArray) -> Union[hv.Layout, hv.NdOverlay]:
    """
    Visualize result of motion estimation.

    This function plot motions across time. If the input has two dimensions,
    they are interpreted as rigid shifts along the "height" and "width"
    dimension of the movie, and plotted as curves across time. If the input has
    more than two dimensions, it is assumed that non-rigid motion estimation was
    enabled and each frame is split into several patches that will each have
    their own shifts. The separate shifts for patches within each frame are
    flattened into a column, then shifts along "height" and "width" dimensions
    are separately plotted as 2d images across time, whose columns represent
    frames and colors represent degree of shift.

    Parameters
    ----------
    motion : xr.DataArray
        Estimated motion.

    Returns
    -------
    Union[hv.Layout, hv.NdOverlay]
        If `motion` contains rigid shifts, then an overlay of two curves are
        returned. Otherwise two images representing non-rigid motions are
        returned.
    """
    if motion.ndim > 2:
        opts_im = {
            "frame_width": 500,
            "aspect": 3,
            "cmap": "RdBu",
            "symmetric": True,
            "colorbar": True,
        }
        mheight = motion.sel(shift_dim="height").stack(grid=["grid0", "grid1"])
        mwidth = motion.sel(shift_dim="width").stack(grid=["grid0", "grid1"])
        mheight = mheight.assign_coords(grid=np.arange(mheight.sizes["grid"]))
        mwidth = mwidth.assign_coords(grid=np.arange(mwidth.sizes["grid"]))
        return (
            (
                hv.Image(mheight.rename("height_motion"), kdims=["frame", "grid"]).opts(
                    title="height_motion", **opts_im
                )
                + hv.Image(mwidth.rename("width_motion"), kdims=["frame", "grid"]).opts(
                    title="width_motion", **opts_im
                )
            )
            .cols(1)
            .opts(show_title=True)
        )
    else:
        opts_cv = {"frame_width": 500, "tools": ["hover"], "aspect": 2}
        return hv.NdOverlay(
            dict(
                width=hv.Curve(motion.sel(shift_dim="width")).opts(**opts_cv),
                height=hv.Curve(motion.sel(shift_dim="height")).opts(**opts_cv),
            )
        )



def add_title(img: Image, title: str, title_size: int) -> Image:
    """
    Adds title on top of image.
    
    Parameters
    ----------
    img: Image
        Image on top of which title will be added.
    title: str
        Text of the title.
    title_size: int
        Font size of the title.
    
    Returns
    -------
    img_new: Image
        Image with added title.
    """
    font = ImageFont.truetype("FONTS/arial.ttf", title_size) 
    _, _, title_width, title_height = ImageDraw.Draw(img).textbbox((0, 0), title, font=font)
    title_pad = title_height // 3

    img_new = Image.new(img.mode, (img.width, img.height + title_height + title_pad), color=(255,255,255))
    img_new.paste(img, (0, title_height + title_pad))
    draw = ImageDraw.Draw(img_new)
    draw.text(((img_new.width - title_width)/2, 0), title, font=font, fill=(0,0,0))
    return img_new
 
 
def screenshot_plot(
    plot,
    title: Optional[str] = None,
    name: Optional[str]=None,
    dest_folder: Union[Path, str] = Path.cwd() / 'screenshots',
    selector: str = '".bk-root .bk"',
) -> Path:
    """
    Exports plot to html and takes its screenshot.
    
    Parameters
    ----------
    plot: Image
        Image on top of which title will be added.
    title: str, optional
        Text of the title. This value is optional. If it is not given image will have
        no title
    name: str, optional
        name of the file. If name is not given, it will be created from title. If neither
        title nor name are given, default name is 'output.png'.
    dest_folder: Path, str
        Default is './screenshots'.
    selector: str
        CSS selector for part of HTML export which will be kept in final export. Default
        is '".bk-root .bk"'.
    
    Returns
    -------
    dest: Path
        Path to raw screenshot which has to be postprocessed. Path is final location of
        saved plot.
    """
    if not name and not title:
        name = 'output.png'
    else:
        name = name if name else title.strip().lower().replace(' ', '_').replace(',', '') \
                                    .replace(':', '').replace('=', '-').replace("'", '') \
                                    .replace('"', ""). replace('{', '').replace('}', '') + '.png'
    dest_folder = Path(dest_folder) if type(dest_folder) is str else dest_folder
    dest_folder.mkdir(parents=True, exist_ok=True)
    tmp_html = dest_folder / '__tmp.html'
    plot_base = type(plot).__module__
    panel_plot = plot_base.startswith('panel')
    holoviews_plot = plot_base.startswith('holoviews')
    if panel_plot:
        plot.save(tmp_html)
    elif holoviews_plot:
        hv.save(plot, tmp_html, fmt='html')
    else:
        print(f'Not implemented: {plot_base}')
        return
    dest = dest_folder / name
    if dest.exists():
        dest.unlink()
    command = f"shot-scraper shot {tmp_html.as_posix()} -o {dest.as_posix()} --selector {selector}"
    success = 0
    repeat = 5
    retry_wait = 1
    downloaded = False
    for _ in range(repeat):
        res = subprocess.call(command, shell=True)
        if res == success and dest.exists():
            downloaded = True
            break
        time.sleep(retry_wait)
    if not downloaded:
        raise RuntimeError('Failed to screenshot html. Potential problem with shot-scraper.'
                           f'Subprocess result: {res=}, file created: {dest.exists()}.')
    tmp_html.unlink()
    return dest


def remove_bokeh_toolbar(
    img_raw: Image,
    toolbar_size: int = 35,
    toolbar_pos: str = 'top',
    comment_height: int = 30,
    comment_present: bool = False,
    crop_bottom: int = 0,
) -> Image:
    """
    Removes Bokeh toolbar from screenshot. If there is a comment/info above horizontally
    orientet toolbar at the top of image, it keeps the comment.
    
    Parameters
    ----------
    img_raw: Image
        Raw image object which needs to be cropped according to the data in other
        parameters.
    toolbar_size: int
        Size (in pixel) of Bokeh toolbar which will be cropped from the screenshot.
        If toolbar_pos is 'top' or 'bottom', this is toolbar height, otherwise it is
        toolbar width.
        Default is 35 px for horisontal toolbar ('left', 'right' orientations) and 25px for
        vertical ('up', 'down' orientations).
    toolbar_pos: str
        Toolbar position related to the rest of plot. Possible values: 'top', 'bottom',
        'left', 'right', 'none'.
    comment_height: int
        Height (in pixel) of comment text which will be kept in the screenshot when
        removing the Bokeh toolbar. It is usually above the toolbar
        Default is 30 px.
    comment_present: bool
        Describes if there is a comment text present above Bokeh toolbar. Default value
        is false.
    crop_bottom: int
        Blank space (in pixel) of plot which will be crop from the final image. Default
        value is 0px.
        
    Returns
    -------
    img: Image
        Path to raw screenshot which has to be postprocessed. Path is final location of
        saved plot.
    """
    toolbar_horizontal = toolbar_pos in ('top', 'down')
    toolbar_vertical = toolbar_pos in ('left', 'right')
    toolbar_size = toolbar_size - 10 if toolbar_vertical else toolbar_size
    img_height = img_raw.height - toolbar_size*toolbar_horizontal - crop_bottom
    img_width = img_raw.width - toolbar_size*toolbar_vertical
    x_0 = toolbar_size*(toolbar_pos == 'left')
    y_0 = toolbar_size*(toolbar_pos == 'top')
    img = Image.new(img_raw.mode, (img_width, img_height), color=(255,255,255))
    if comment_present:
        box_comment = (0, 0, img.width, comment_height)
        img_comment = img_raw.crop(box_comment)
        img.paste(img_comment, (0, 0))
    box_plot = (x_0, y_0 + comment_height*comment_present,
                x_0 + img_width, y_0 + img_height + comment_height*comment_present)
    img_plot = img_raw.crop(box_plot)
    img.paste(img_plot, (0, comment_height*comment_present))
    return img

 
def export_plot(
    plot,
    title: Optional[str] = None,
    title_size: int = 25,
    name: Optional[str]=None,
    dest_folder: Union[Path, str] = Path.cwd() / 'screenshots',
    selector: str = '".bk-root .bk"',
    toolbar_size: int = 35,
    toolbar_pos: str = 'top',
    comment_height: int = 30,
    comment_present: bool = False,
    crop_bottom: int = 0,
):
    """
    Exports plot to the image. Image is saved according to the parameter `dest_folder`,
    named according to the parameters `name` or `title`.
    
    Parameters
    ----------
    plot: Image
        Image on top of which title will be added.
    title: str, optional
        Text of the title. This value is optional. If it is not given image will have
        no title
    title_size: int
        Font size of the title.
    name: str, optional
        name of the file. If name is not given, it will be created from title. If neither
        title nor name are given, default name is 'output.png'.
    dest_folder: Path, str
        Default is './screenshots'.
    selector: str
        CSS selector for part of HTML export which will be kept in final export. Default
        is '".bk-root .bk"'.
    toolbar_size: int
        Size (in pixel) of Bokeh toolbar which will be cropped from the screenshot.
        If toolbar_pos is 'top' or 'bottom', this is toolbar height, otherwise it is
        toolbar width.
        Default is 35 px for horisontal toolbar ('left', 'right' orientations) and 25px for
        vertical ('up', 'down' orientations).
    toolbar_pos: str
        Toolbar position related to the rest of plot. Possible values: 'top', 'bottom',
        'left', 'right', 'none'.
    comment_height: int
        Height (in pixel) of comment text which will be kept in the screenshot when
        removing the Bokeh toolbar. It is usually above the toolbar
        Default is 30 px.
    comment_present: bool
        Describes if there is a comment text present above Bokeh toolbar. Default value
        is false.
    crop_bottom: int
        Blank space (in pixel) of plot which will be crop from the final image. Default
        value is 0px.
    """
    dest = screenshot_plot(plot, title=title, name=name, dest_folder=dest_folder, selector=selector)
    img = Image.open(dest)
    img = remove_bokeh_toolbar(img, toolbar_size=toolbar_size, toolbar_pos=toolbar_pos,
                    comment_height=comment_height, comment_present=comment_present,
                    crop_bottom=crop_bottom) 
    if title:
        img = add_title(img, title, title_size) 
    img.save(dest)
    print(f'Plot succesfully saved at {str(dest)}.')
