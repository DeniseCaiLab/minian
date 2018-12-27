import functools as fct
import numpy as np
import holoviews as hv
import xarray as xr
import ipywidgets as iwgt
import seaborn as sns
import pandas as pd
import colorsys
import param
import dask.array as da
from .utilities import scale_varr
from .motion_correction import shift_fft
from collections import OrderedDict
from holoviews.streams import Stream, Pipe, RangeXY, DoubleTap, Tap, Selection1D, BoxEdit
from holoviews.operation import contours, threshold
from holoviews.operation.datashader import datashade, regrid, dynspread
from holoviews.util import Dynamic
from datashader.colors import Sets1to3
from datashader import count_cat
from datashader import mean as dsmean
from datashader import count as dscount
from IPython.core.display import display, clear_output
from bokeh.io import push_notebook, show
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Slider, Range1d, LinearAxis
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.palettes import Category10_10
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage.measurements import center_of_mass
from IPython.core.debugger import set_trace
from dask.diagnostics import ProgressBar
from skvideo.io import FFmpegWriter, vwrite
from scipy import linalg
from bokeh import models
from bokeh.io import export_svgs

# def update_override(self, **kwargs):
#     self._set_stream_parameters(**kwargs)
#     transformed = self.transform()
#     if transformed:
#         self._set_stream_parameters(**transformed)
#     self.trigger([self])


class VArrayViewer():
    def __init__(self, varr, framerate=30, rerange=None, compute=True, datashading=True):
        if isinstance(varr, list):
            self.ds = xr.merge(varr)
        else:
            self.ds = varr.to_dataset()
        # self.ds = hv.Dataset(varr)
        self.mean = self.ds.mean(['height', 'width'])
        self.max = self.ds.max(['height', 'width'])
        self.min = self.ds.min(['height', 'width'])
        self.framerate = framerate
        self._f = self.ds.coords['frame'].values
        self._h = self.ds.sizes['height']
        self._w = self.ds.sizes['width']
        self.rerange = rerange
        CStream = Stream.define(
            'CStream',
            f=param.Integer(default=0, bounds=(self._f.min(), self._f.max())))
        self.stream = CStream()
        self.widgets = self._widgets()
        self._compute = compute
        self._datashade = datashading
        if compute:
            print("computing summary")
            with ProgressBar():
                self.mean = self.mean.compute()
                self.max = self.max.compute()
                self.min = self.min.compute()

    def show(self):
        imdict = OrderedDict()
        meandict = OrderedDict()
        for vname, varr in self.ds.items():
            fim = fct.partial(self._img, dat=varr)
            im = hv.DynamicMap(fim, streams=[self.stream])
            im = regrid(im, height=self._h, width=self._w)
            im = im(plot={'width': self._w, 'height': self._h})
            if self.rerange:
                im = im.redim.range(**{vname: self.rerange})
            xyrange = RangeXY(source=im)
            xyrange = xyrange.rename(x_range='w', y_range='h')
            fhist = fct.partial(self._hist, dat=varr)
            hist = hv.DynamicMap(fhist, streams=[self.stream, xyrange])
            hist = hist(plot={'width': int(np.around(self._w * 0.35)), 'height': self._h})
            if self._compute:
                cur_mdict = OrderedDict()
                dmean = hv.Curve(self.mean, kdims='frame', vdims=vname)
                cur_mdict['mean'] = dmean(plot={'tools': ['hover']})
                dmax = hv.Curve(self.max, kdims='frame', vdims=vname)
                cur_mdict['max'] = dmax(plot={'tools': ['hover']})
                dmin = hv.Curve(self.min, kdims='frame', vdims=vname)
                cur_mdict['min'] = dmin(plot={'tools': ['hover']})
                mean = hv.NdOverlay(cur_mdict, kdims=['variable'])
                if self._datashade:
                    mean = datashade_ndcurve(mean)
                vl = hv.DynamicMap(lambda f: hv.VLine(f), streams=[self.stream])
                vl = vl(style={'color': 'red'})
                mean = (mean * vl).relabel(group=vname, label='Mean')
                mean = mean.opts(
                    plot={'width': self._w, 'height': int(np.around(self._h * 0.6))})
                meandict[vname] = mean
            image = im.relabel(group=vname, label='Image')
            histtogram = hist.relabel(group=vname, label='Histogram')
            imdict[vname] = image << histtogram
        return hv.Layout(list(imdict.values()) + list(meandict.values())).cols(
            len(self.ds))

    def _widgets(self):
        w_paly = iwgt.Play(
            value=self._f[0],
            min=self._f[0],
            max=self._f[-1],
            interval=1000 / self.framerate)
        w_frame = iwgt.SelectionSlider(
            value = self._f[0],
            options = self._f.tolist(),
            continuous_update=False,
            description="Frame:"
        )
        iwgt.link((w_paly, 'value'), (w_frame, 'value'))
        iwgt.interactive(self.stream.event, f=w_frame)
        return iwgt.HBox([w_paly, w_frame])

    def _img(self, dat, f):
        return hv.Image(dat.sel(frame=f).compute(), kdims=['width', 'height'])

    def _hist(self, dat, f, w, h):
        if w and h:
            im = self._img(dat=dat, f=f).select(frame=f, height=h, width=w)
        else:
            im = self._img(dat=dat, f=f)
        return hv.operation.histogram(im, frequency_label='freq', num_bins=50)


class MCViewer():
    def __init__(self, varr, marr=None, framerate=30):
        if isinstance(varr, list):
            varr = xr.merge(varr)
        self.varr = hv.Dataset(varr)
        if marr is not None:
            self.marr = hv.Dataset(marr)
        else:
            self.marr = None
        self.framerate = framerate
        CStream = Stream.define(
            'CStream',
            f=param.Integer(default=0, bounds=self.varr.range('frame')))
        self.stream = CStream()
        self.widgets = self._widgets()

    def show(self, use_datashade=False):
        vh = self.varr.range('height')
        vw = self.varr.range('width')
        him = int(vh[1] - vh[0] + 1)
        wim = int(vw[1] - vw[0] + 1)
        if self.marr is not None:
            mh = self.marr.range('height')
            mw = self.marr.range('width')
            hma = int(mh[1] - mh[0] + 1)
            wma = int(mw[1] - mw[0] + 1)
        varrdict = OrderedDict()
        for dim in self.varr.vdims:
            cur_d = self.varr.reindex(vdims=[dim.name])
            fim = fct.partial(self._img, dat=cur_d)
            im = hv.DynamicMap(fim, streams=[self.stream])
            im = regrid(im, height=him, width=wim)
            im = im(plot={'width': wim, 'height': him})
            image = im.relabel(group=dim.name, label='Image')
            varrdict[dim.name] = image
        if self.marr is not None:
            fma = fct.partial(self._rgb, dat=self.marr)
            ma = hv.DynamicMap(fma, streams=[self.stream])
            ma = regrid(ma, height=hma, width=wma)
            ma = ma.relabel(label="Match Image")
            ma = ma(plot={'height': hma, 'width': wma})
        else:
            ma = hv.Empty()
        return hv.Layout(ma + hv.NdLayout(varrdict, kdims=['name'])).cols(1)

    def _widgets(self):
        dfrange = self.varr.range('frame')
        w_frame = iwgt.IntSlider(
            value=0,
            min=dfrange[0],
            max=dfrange[1],
            continuous_update=False,
            description="Frame:")
        w_paly = iwgt.Play(
            value=0,
            min=dfrange[0],
            max=dfrange[1],
            interval=1000 / self.framerate)
        iwgt.jslink((w_paly, 'value'), (w_frame, 'value'))
        iwgt.interactive(self.stream.event, f=w_frame)
        return iwgt.HBox([w_paly, w_frame])

    def _img(self, dat, f):
        return hv.Image(dat.select(frame=f), kdims=['width', 'height'])

    def _rgb(self, dat, f):
        return hv.RGB(dat.select(frame=f), kdims=['width', 'height'])


class CNMFViewer():
    def __init__(self, cnmf, Y, framerate=30):
        self.cnmf = cnmf
        # self.cnmf_vld = cnmf.sel(unit_id=cnmf.attrs['unit_mask'])
        self.cnmf_vld = cnmf
        # self.ds = hv.Dataset(cnmf)
        # self.ds_vld = hv.Dataset(self.cnmf_vld)
        self.framerate = framerate
        self.Y = Y
        # _rh = self.ds_vld.range('height')
        # _rw = self.ds_vld.range('width')
        # self._h = int(_rh[1] - _rh[0])
        # self._w = int(_rw[1] - _rw[0])
        self._h = self.cnmf_vld.sizes['height']
        self._w = self.cnmf_vld.sizes['width']
        self._f = self.cnmf_vld.sizes['frame']
        self._u = self.cnmf.sizes['unit_id']
        self._update_mov = False
        self.strm_f = DoubleTap(x=0).rename(x='f', y=None)
        self.strm_f.add_subscriber(self._update_f)
        self.strm_uid = Selection1D()
        self.pipY = Pipe(data=[])
        self.pipAdC = Pipe(data=[])
        self.pipbdf = Pipe(data=[])
        self.pipYr = Pipe(data=[])
        self._cur_sel = (0, 5)
        self._overlay = True
        self.widgets = self._widgets()
        self.hvobjs = self.get_plot()

    def get_plot(self):
        cur_sel = (self._cur_sel[0], self._cur_sel[1])
        cur_units = self.cnmf_vld.isel(
            unit_id=slice(*cur_sel)).coords['unit_id'].values
        cont_dict = OrderedDict()
        A = self.cnmf_vld['A']
        for uid in self.cnmf_vld.coords['unit_id']:
            cur_A = A.sel(unit_id=uid).load()
            cur_thres = float(cur_A.max() * 0.3)
            cur_cent = center_of_mass(cur_A.data)
            cur_im = hv.Image(cur_A, kdims=['width', 'height'])
            cur_cont = contours(cur_im, levels=[cur_thres])
            cur_cont = cur_cont(
                plot={
                    'show_legend': False,
                    'tools': ['hover', 'box_select', 'lasso_select', 'tap']
                },
                style={'cmap': ['white']})
            cur_text = hv.Text(cur_cent[1], cur_cent[0], str(int(uid)))
            cur_text = cur_text(style={
                'color': 'white',
                'text_font_size': '8pt'
            })
            cont_dict[int(uid)] = cur_cont * cur_text
        ovly = hv.NdOverlay(cont_dict, kdims=['unit_id'])
        self.strm_uid.source = ovly
        fim = fct.partial(hv.Image, kdims=['width', 'height'])
        img_Y = hv.DynamicMap(fim, streams=[self.pipY])
        img_Y = regrid(
            img_Y, height=int(self._h / 10), width=int(self._w / 10))
        img_Y = img_Y(plot={
            'height': self._h,
            'width': self._w,
            'title_format': "Y (Original)"
        })
        img_AdC = hv.DynamicMap(fim, streams=[self.pipAdC])
        img_AdC = regrid(
            img_AdC, height=int(self._h / 10), width=int(self._w / 10))
        img_AdC = img_AdC(plot={
            'height': self._h,
            'width': self._w,
            'title_format': "A dot C (Units)"
        })
        img_bdf = hv.DynamicMap(fim, streams=[self.pipbdf])
        img_bdf = regrid(
            img_bdf, height=int(self._h / 10), width=int(self._w / 10))
        img_bdf = img_bdf(
            plot={
                'height': self._h,
                'width': self._w,
                'title_format': "b dot f (Background)"
            })
        img_Yr = hv.DynamicMap(fim, streams=[self.pipYr])
        img_Yr = regrid(
            img_Yr, height=int(self._h / 10), width=int(self._w / 10))
        img_Yr = img_Yr(plot={
            'height': self._h,
            'width': self._w,
            'title_format': "Yr (Residule)"
        })
        if self._overlay:
            nunits = len(cur_units)
            cur_A = scale_varr(self.cnmf_vld['A'].sel(unit_id=cur_units))
            cur_C = scale_varr(self.cnmf_vld['C'].sel(unit_id=cur_units))
            cur_A = cur_A.load()
            cur_C = cur_C.load()
            clr_rgb = sns.color_palette('hsv', nunits)
            clr_rgb_xr = xr.DataArray(
                clr_rgb,
                coords={
                    'unit_id': cur_A.coords['unit_id'],
                    'cspace': ['R', 'G', 'B']
                },
                dims=['unit_id', 'cspace'])
            im_rgb = (cur_A.dot(clr_rgb_xr) / cur_A.sum('unit_id')).fillna(0)
            hsv_coords = im_rgb.coords
            hsv_coords['cspace'] = ['H', 'S', 'V']
            im_hsv = xr.DataArray(
                im_rgb.values, coords=hsv_coords, dims=im_rgb.dims)
            im_hsv.values = rgb_to_hsv(im_hsv.values)
            fim = fct.partial(
                self._im_overlay, A=cur_A, C=cur_C, im_hsv=im_hsv)
            imgs_pu = hv.DynamicMap(fim, streams=[self.strm_f])
            imgs_pu = regrid(
                imgs_pu, height=int(self._h / 10), width=int(self._w / 10))
            # imgs_pu = regrid(imgs_pu, x_sampling=4, y_sampling=4)
            imgs_pu = imgs_pu(plot={
                'height': self._h,
                'width': self._w,
                'title_format': "overlay - A"
            }) * ovly
            c_pu = OrderedDict()
            for uid in cur_units:
                cc = hv.Curve(
                    self.cnmf_vld.sel(unit_id=uid)['C'], kdims=['frame'])
                cc = cc(
                    plot={
                        'height': self._h,
                        'width': self._w * 2,
                        'title_format': "overlay - C",
                        'tools': []
                    },
                    style={'color': hv.Cycle(values=clr_rgb)})
                c_pu[uid] = cc
            c_pu = hv.NdOverlay(c_pu, kdims=['unit_id'])
            self.strm_f.source = c_pu
            vl = hv.DynamicMap(lambda f: hv.VLine(f), streams=[self.strm_f])
            vl = vl(style={'color': 'red'})
            c_pu = c_pu * vl
        else:
            imgs_pu = OrderedDict()
            c_pu = OrderedDict()
            for uid in cur_units:
                im = hv.Image(
                    self.cnmf_vld.sel(unit_id=uid)['A'],
                    kdims=['width', 'height'])
                im = regrid(im)
                im = im(plot={'height': self._h, 'width': self._w})
                imgs_pu[uid] = im
                cc = hv.Curve(
                    self.cnmf_vld.sel(unit_id=uid)['C'], kdims=['frame'])
                cs = hv.Curve(
                    self.cnmf_vld.sel(unit_id=uid)['S'], kdims=['frame'])
                cs = cs(
                    plot={
                        'finalize_hooks': [self._twinx],
                        'apply_ranges': False
                    },
                    style={'color': 'red'})
                vl = hv.DynamicMap(
                    lambda f: hv.VLine(f), streams=[self.strm_f])
                vl = vl(style={'color': 'red'})
                c = cc * vl
                c = c(plot={'height': self._h, 'width': self._w * 2})
                c_pu[uid] = c
            imgs_pu = hv.NdLayout(imgs_pu, kdims=['unit_id']).cols(1)
            c_pu = hv.NdLayout(c_pu, kdims=['unit_id']).cols(1)
        hvobjs = (img_Y + img_AdC + img_bdf + img_Yr + imgs_pu + c_pu).cols(2)
        self.hvobjs = hvobjs
        return hvobjs

    def show(self):
        display(self.widgets)
        display(self.hvobjs)

    def _set_sel(self, change):
        self._cur_sel = change['new']

    def _set_update(self, change):
        self._update_mov = change['new']
        if change['new']:
            self._update_f(f=self.strm_f.contents['f'])

    def _set_overlay(self, change):
        self._overlay = change['new']

    def _update_plot(self, b):
        clear_output()
        self.get_plot()
        self.show()

    def _update_f(self, f):
        if f is not None:
            f = int(f)
            if self._update_mov:
                self.pipY.send([])
                self.pipAdC.send([])
                self.pipbdf.send([])
                self.pipYr.send([])
                cur_Y = self.Y.sel(frame=f)
                cur_AdC = self.cnmf_vld['A'].dot(
                    self.cnmf_vld['C'].sel(frame=f))
                cur_bdf = self.cnmf_vld['b'].dot(
                    self.cnmf_vld['f'].sel(frame=f))
                cur_Yr = cur_Y - cur_AdC - cur_bdf
                self.pipY.send(cur_Y)
                self.pipAdC.send(cur_AdC)
                self.pipbdf.send(cur_bdf)
                self.pipYr.send(cur_Yr)

    def _f_vl(self, f):
        if f is not None:
            self._update_f(f)
            return hv.VLine(f)
        else:
            return hv.VLine(0)

    def _im_overlay(self, f, A, C, im_hsv, contour=None):
        f = int(f)
        AdC = A.dot(C.sel(frame=f))
        im_hue = im_hsv.sel(cspace='H').rename('H').drop('cspace')
        im_sat = (im_hsv.sel(cspace='S')).rename('S').drop('cspace')
        im_val = (im_hsv.sel(cspace='V') * AdC * 4).clip(
            0, 1).rename('V').drop('cspace')
        ds = xr.merge([im_hue, im_sat, im_val])
        im = hv.HSV(ds, kdims=['width', 'height'])
        # if contour is None:
        #     contour = hv.operation.contours(im)
        return im

    def _twinx(self, plot, element):
        # Setting the second y axis range name and range
        start, end = (element.range(1))
        label = element.dimensions()[1].pprint_label
        plot.state.extra_y_ranges = {"foo": Range1d(start=start, end=end)}
        # Adding the second axis to the plot.
        linaxis = LinearAxis(axis_label=label, y_range_name='foo')
        plot.state.add_layout(linaxis, 'right')

    def _widgets(self):
        dfrange = [0, self._f]
        w_frame = iwgt.IntSlider(
            value=0,
            min=dfrange[0],
            max=dfrange[1],
            continuous_update=False,
            description="Frame:",
            layout=iwgt.Layout(width='50%'))
        w_play = iwgt.Play(
            value=0,
            min=dfrange[0],
            max=dfrange[1],
            interval=1000 / self.framerate)
        iwgt.jslink((w_play, 'value'), (w_frame, 'value'))
        iwgt.interactive(self.strm_f.event, x=w_frame)
        uidrange = [0, self._u]
        w_select = iwgt.IntRangeSlider(
            value=self._cur_sel,
            min=uidrange[0],
            max=uidrange[1],
            continuous_update=False,
            description="Unit ID:",
            layout=iwgt.Layout(width='50%'))
        w_select.observe(self._set_sel, names='value')
        w_update = iwgt.Button(description="Update")
        w_update.on_click(self._update_plot)
        w_update_mov = iwgt.Checkbox(
            value=self._update_mov, description="Update Movies")
        w_update_mov.observe(self._set_update, names='value')
        w_overlay = iwgt.Checkbox(value=self._overlay, description="Overlay")
        w_overlay.observe(self._set_overlay, names='value')
        return iwgt.VBox([
            iwgt.HBox([w_frame, w_play, w_update_mov]),
            iwgt.HBox([w_select, w_update, w_overlay])
        ])


class AlignViewer():
    def __init__(self, shiftds, sampling=2):
        self.shiftds = shiftds
        self.temps = shiftds['temps']
        self.mask = xr.zeros_like(self.temps, dtype=bool)
        self.ls_anm = np.unique(shiftds.coords['animal'].values)
        self.ls_ss = np.unique(shiftds.coords['session'].values)
        Selection = Stream.define(
            'selection',
            anm=param.Selector(self.ls_anm),
            ss=param.Selector(self.ls_ss))
        self.str_sel = Selection(anm=self.ls_anm[0], ss=self.ls_ss[0])
        self.sampling = sampling
        self.str_box = BoxEdit()
        self.box = hv.DynamicMap(self._box, streams=[self.str_box])
        self.box = self.box.opts(
            style=dict(fill_alpha=0.3, line_color='white'))
        self.wgts = self._widgets()
        self.hvobjs = self._get_objs()

    def show(self):
        display(self.wgts)
        display(self.hvobjs)

    def _box(self, data):
        if data is None:
            return hv.Polygons([])
        else:
            diag = pd.DataFrame(data)
            corners = []
            for ir, r in diag.iterrows():
                corners.append(
                    np.array([(r['x0'], r['y0']), (r['x0'], r['y1']),
                              (r['x1'], r['y1']), (r['x1'], r['y0'])]))
            return hv.Polygons([{
                ('x', 'y'): cnr.squeeze()
            } for cnr in corners])

    def _temps(self, anm, ss):
        ims = hv.Image(
            self.temps.sel(animal=anm, session=ss), ['width', 'height'])
        return ims

    def _temp0(self, anm, ss):
        im0 = hv.Image(
            self.temps.sel(animal=anm).isel(session=0), ['width', 'height'])
        return im0

    def _re(self, anm, ss):
        re = hv.Image(self.shiftds['temps_shifted'].sel(
            animal=anm, session=ss), ['width', 'height'])
        return re

    def _widgets(self):
        sel_anm = iwgt.Dropdown(options=self.ls_anm, description='Animal:')
        sel_ss = iwgt.Dropdown(options=self.ls_ss, description='Session:')
        bt_mask0 = iwgt.Button(description="Update Template")
        bt_mask = iwgt.Button(description="Update Target")
        bt_mask0.on_click(self._save_mask0)
        bt_mask.on_click(self._save_mask)
        iwgt.interactive(self.str_sel.event, anm=sel_anm, ss=sel_ss)
        return iwgt.HBox([sel_anm, sel_ss, bt_mask0, bt_mask])

    def _get_objs(self):
        im0 = regrid(hv.DynamicMap(self._temp0, streams=[self.str_sel]))
        ims = regrid(hv.DynamicMap(self._temps, streams=[self.str_sel]))
        re = regrid(hv.DynamicMap(self._re, streams=[self.str_sel]))
        return ims * self.box.clone(link_inputs=False) + im0 * self.box + re

    def _save_mask0(self, _):
        print("save mask 0")
        cur_anm = self.str_sel.anm
        cur_ss = self.str_sel.ss
        h0, hd = self.str_box.data['y0'][-1], np.round(
            self.str_box.data['y1'][-1] - self.str_box.data['y0'][-1])
        w0, wd = self.str_box.data['x0'][-1], np.round(
            self.str_box.data['x1'][-1] - self.str_box.data['x0'][-1])
        self.mask[dict(session=0)].loc[dict(animal=self.str_sel.anm)] = False
        self.mask[dict(session=0)].loc[dict(
            animal=self.str_sel.anm,
            height=slice(h0, h0 + hd),
            width=slice(w0, w0 + wd))] = True

    def _save_mask(self, _):
        print("entering")
        cur_anm = self.str_sel.anm
        cur_ss = self.str_sel.ss
        h1, hd = self.str_box.data['y0'][-1], np.round(
            self.str_box.data['y1'][-1] - self.str_box.data['y0'][-1])
        w1, wd = self.str_box.data['x0'][-1], np.round(
            self.str_box.data['x1'][-1] - self.str_box.data['x0'][-1])
        self.mask.loc[dict(animal=self.str_sel.anm,
                           session=self.str_sel.ss)] = False
        self.mask.loc[dict(
            animal=self.str_sel.anm,
            session=self.str_sel.ss,
            height=slice(h1, h1 + hd),
            width=slice(w1, w1 + wd))] = True
        temp_src = self.temps.sel(animal=cur_anm).isel(session=0)
        mask_src = self.mask.sel(animal=cur_anm).isel(session=0)
        temp_dst = self.temps.sel(animal=cur_anm, session=cur_ss)
        mask_dst = self.mask.sel(animal=cur_anm, session=cur_ss)
        temp_src = temp_src.where(mask_src, drop=True)
        temp_dst = temp_dst.where(mask_dst, drop=True)
        man_sh_w = temp_src.coords['width'][0].values - temp_dst.coords['width'][0].values
        man_sh_h = temp_src.coords['height'][0].values - temp_dst.coords['height'][0].values
        man_sh = xr.DataArray(
            [man_sh_w, man_sh_h],
            coords=dict(shift_dim=['width', 'height']),
            dims=['shift_dim'])
        print("done selection")
        temp_src_fft = np.fft.fft2(temp_src)
        temp_dst_fft = np.fft.fft2(temp_dst)
        cur_res = shift_fft(temp_src_fft, temp_dst_fft)
        cur_sh = cur_res[0:2]
        cur_cor = cur_res[2]
        cur_sh = xr.DataArray(
            np.round(cur_sh),
            coords=dict(shift_dim=list(temp_dst.dims)),
            dims=['shift_dim'])
        cur_sh = cur_sh + man_sh
        sh_dict = cur_sh.astype(int).to_series().to_dict()
        print(sh_dict)
        self.shiftds['temps_shifted'].loc[dict(
            animal=cur_anm, session=cur_ss)] = self.temps.sel(
                animal=cur_anm, session=cur_ss).shift(**sh_dict)
        self.shiftds['shifts'].loc[dict(
            animal=cur_anm, session=cur_ss)] = cur_sh
        self.shiftds['corrs'].loc[dict(
            animal=cur_anm, session=cur_ss)] = cur_cor

        
def generate_videos(minian, vpath, chk=None, pre_compute=False):
    print("generating traces")
    if not chk:
        chk = dict(height='auto', width='auto', frame='auto')
    A = minian['A'].chunk(dict(height=chk['height'], width=chk['width'], unit_id=-1))
    C = minian['C'].chunk(dict(frame=chk['frame'], unit_id=-1))
    Y = minian['Y'].chunk(dict(frame=chk['frame'], height=chk['height'], width=chk['width']))
    try:
        B = minian['B'].chunk(dict(unit_id=-1))
    except KeyError:
        print("cannot find background term")
        B = 0
    org = minian['org'].chunk(dict(frame=chk['frame'], height=chk['height'], width=chk['width']))
    C = C + B
    AC = xr.apply_ufunc(
        da.dot, A, C,
        input_core_dims=[['height', 'width', 'unit_id'], ['unit_id', 'frame']],
        output_core_dims=[['height', 'width', 'frame']],
        dask='allowed',
        output_dtypes=[Y.dtype])
    res = scale_varr(org, pre_compute=pre_compute) - scale_varr(AC, pre_compute=pre_compute)
    org_norm = scale_varr(org, (0, 255), pre_compute=pre_compute).astype(np.uint8)
    Y_norm = scale_varr(Y, (0, 255), pre_compute=pre_compute).astype(np.uint8)
    AC_norm = scale_varr(AC, (0, 255), pre_compute=pre_compute).astype(np.uint8)
    res_norm = scale_varr(res, (0, 255), pre_compute=pre_compute).astype(np.uint8)
#     with ProgressBar():
#         Y_norm = Y_norm.compute()
#         AC_norm = AC_norm.compute()
#         res_norm = res_norm.compute()
    vid = xr.concat([
        xr.concat([org_norm, Y_norm], 'width'),
        xr.concat([AC_norm, res_norm], 'width')],
        dim='height')
    print("writing videos")
    with ProgressBar():
        vwrite(vpath, vid.transpose('frame', 'height', 'width').values)
#     vwrt = FFmpegWriter(vpath)
#     for fid, fm in vid.rolling(frame=1):
#         print("writing frame {}".format(fid.values), end='\r')
#         vwrite(vpath, fm.values)
#     vwrt.close()


def datashade_ndcurve(ovly, kdim=None):
    if not kdim:
        kdim = ovly.kdims[0].name
    var = np.unique(ovly.dimension_values(kdim)).tolist()
    color_key = [(v, Category10_10[iv]) for iv, v in enumerate(var)]
    color_pts = hv.NdOverlay(
        {k: hv.Points([0, 0], label=str(k)).opts(style=dict(color=v)) for k, v in color_key})
    ds_ovly = dynspread(datashade(
        ovly,
        aggregator=count_cat(kdim),
        color_key=dict(color_key), min_alpha=128, normalization='eq_hist'))
    return ds_ovly * color_pts


def construct_G(g, T):
    cur_c, cur_r = np.zeros(T), np.zeros(T)
    cur_c[0] = 1
    cur_r[0] = 1
    cur_c[1:len(g) + 1] = -g
    return linalg.toeplitz(cur_c, cur_r)

def normalize(a): return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, +1))

def convolve_G(s, g):
    G = construct_G(g, len(s))
    try:
        c = linalg.inv(G).dot(s)
    except np.linalg.LinAlgError:
        c = s.copy()
    return c

def construct_pulse_response(g, length=500):
    s = np.zeros(length)
    s[np.arange(0, length, 500)] = 1
    c = convolve_G(s, g)
    return s, c


def centroid(A):
    def rel_cent(im):
        cent = np.array(center_of_mass(im))
        return cent / im.shape
    cents = (xr.apply_ufunc(
        rel_cent, A.chunk(dict(height=-1, width=-1)),
        input_core_dims=[['height', 'width']],
        output_core_dims=[['dim']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float],
        output_sizes=dict(dim=2))
             .assign_coords(dim=['height', 'width']))
    cents_df = (cents.rename('cents').to_series().unstack()
               .reset_index().rename_axis(None, axis='columns'))
    cents_df['height'] *= A.coords['height'].max().values
    cents_df['width'] *= A.coords['width'].max().values
    meta_df = (cents.rename('cents').to_dataframe().reset_index()
               .drop(['dim', 'cents'], axis='columns').drop_duplicates())
    cents_df = cents_df.merge(meta_df, on='unit_id')
    return cents_df


def visualize_seeds(Y, seeds):
    pass


def visualize_spatial_update(A_dict, C_dict, kdims=None):
    if not kdims:
        A_dict = dict(dummy=A_dict)
        C_dict = dict(dummy=C_dict)
    hv_pts_dict, hv_A_dict, hv_Ab_dict, hv_C_dict = (
        dict(), dict(), dict(), dict())
    for key, A in A_dict.items():
        A = A.compute()
        C = C_dict[key].compute()
        h, w = A.sizes['height'], A.sizes['width']
        cents_df = centroid(A)
        hv_pts_dict[key] = (hv.Points(cents_df,
                                      kdims=['width', 'height'],
                                      vdims=['unit_id'])
                            .opts(plot=dict(tools=['hover']),
                                  style=dict(fill_alpha=0.2,
                                             line_alpha=0, size=8)))
        hv_A_dict[key] = hv.Image(A.sum('unit_id').rename('A'),
                                  kdims=['width', 'height'])
        hv_Ab_dict[key] = hv.Image((A > 0).sum('unit_id').rename('A_bin'),
                                   kdims=['width', 'height'])
        hv_C_dict[key] = hv.Dataset(C.rename('C')).to(hv.Curve, kdims='frame')
    cropts = {'plot': {
        'height': int(np.around(0.13 * w)),
        'width': int(np.around(1.8 * w))}}
    hv_pts = hv.HoloMap(hv_pts_dict, kdims=kdims)
    hv_A = (regrid(hv.HoloMap(hv_A_dict, kdims=kdims))
            .opts(plot=dict(height=h, width=w, colorbar=True),
                  style=dict(cmap='Viridis')))
    hv_Ab = (regrid(hv.HoloMap(hv_Ab_dict, kdims=kdims))
            .opts(plot=dict(height=h, width=w, colorbar=True),
                  style=dict(cmap='Viridis')))
    hv_C = (datashade(hv.HoloMap(hv_C_dict, kdims=kdims).collate()
                      .grid('unit_id').add_dimension('time', 0, 0),
                      min_alpha=128)
            .map(lambda cr: cr.opts(**cropts), hv.RGB))
    return ((hv_pts * hv_A).relabel('Spatial Matrix')
            + (hv_pts * hv_Ab).relabel('Binary Spatial Matrix')
            + hv_C.relabel('Temporal Components')
            + hv.Div('')).cols(2)


def visualize_temporal_update(YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict, kdims=None, norm=True, datashading=True):
    inputs = [YA_dict, C_dict, S_dict, sig_dict, g_dict]
    if not kdims:
        inputs = [dict(dummy=i) for i in inputs]
        A_dict = dict(dummy=A_dict)
    input_dict = {k: [i[k] for i in inputs] for k in inputs[0].keys()}
    hv_YA, hv_C, hv_S, hv_sig, hv_C_pul, hv_S_pul, hv_A = [dict() for _ in range(7)]
    for k, ins in input_dict.items():
        ins[0] = ins[0].sel(unit_id=ins[1].coords['unit_id'])
        if norm:
            ins[:-1] = [xr.apply_ufunc(
                normalize, i.chunk(dict(frame=-1, unit_id='auto')),
                input_core_dims=[['frame']],
                output_core_dims=[['frame']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[i.dtype]
            ) for i in ins[:-1]]
        ins[:] = [i.compute() for i in ins]
        ya, c, s, sig, g = ins
        f_crd = ya.coords['frame']
        s_pul, c_pul = xr.apply_ufunc(
            construct_pulse_response, g,
            input_core_dims=[['lag']],
            output_core_dims=[['t'], ['t']],
            vectorize=True,
            # kwargs=dict(length=len(f_crd)),
            output_sizes=dict(t=500))
        s_pul, c_pul = (s_pul.assign_coords(t=f_crd.values[:500]),
                        c_pul.assign_coords(t=f_crd.values[:500]))
        pul_range = (
            f_crd.min(),
            int(np.around(f_crd.min() + (f_crd.max() - f_crd.min()) / 2)))
        hv_S_pul[k], hv_C_pul[k] = [
            (hv.Dataset(tr.rename('Response (A.U.)'))
             .to(hv.Curve, kdims=['t'])) for tr in [s_pul, c_pul]]
        hv_YA[k], hv_C[k], hv_S[k], hv_sig[k] = [
            (hv.Dataset(tr.rename('Intensity (A.U.)'))
             .to(hv.Curve, kdims=['frame'])) for tr in [ya, c, s, sig]]
        hv_A[k] = (hv.Dataset(A_dict[k].rename('A'))
                   .to(hv.Image, kdims=['width', 'height']))
        h, w = A_dict[k].sizes['height'], A_dict[k].sizes['width']
    hvobjs = [hv_YA, hv_C, hv_S, hv_sig, hv_C_pul, hv_S_pul, hv_A]
    hvobjs[:] = [hv.HoloMap(hvobj, kdims=kdims).collate() for hvobj in hvobjs]
    hv_unit = {'Raw Signal': hvobjs[0], 'Fitted Calcium Trace': hvobjs[1],
               'Fitted Spikes': hvobjs[2], 'Fitted Signal': hvobjs[3]}
    hv_pul = {'Simulated Calcium': hvobjs[4], 'Simulated Spike': hvobjs[5]}
    hv_unit = hv.HoloMap(hv_unit, kdims='traces').collate().overlay('traces')
    hv_pul = hv.HoloMap(hv_pul, kdims='traces').collate().overlay('traces')
    if datashading:
        hv_unit = datashade_ndcurve(hv_unit, 'traces')
        hv_pul = datashade_ndcurve(hv_pul, 'traces')
        hv_A = regrid(hvobjs[6])
    else:
        hv_unit = hv.DynamicMap(hv_unit)
        hv_pul = hv.DynamicMap(hv_pul)
        hv_A = hv.DynamicMap(hvobjs[6])
    hv_unit = hv_unit.opts(plot=dict(height=h, width=2*w))
    hv_pul = (hv_pul.opts(plot=dict(height=h, width=w))
              .redim(t=hv.Dimension('t', soft_range=pul_range)))
    hv_A = hv_A.opts(plot=dict(height=h, width=w), style=dict(cmap='Viridis'))
    return (hv_unit.relabel("Temporal Traces")
            + hv.Div('')
            + hv_pul.relabel("Simulated Pulse Response")
            + hv_A.relabel("Spatial Footprint")).cols(2)


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def _get_figures_core(objs):
    if isinstance(objs, list):
        objs = [_get_figures_core(plot) for plot in objs]
    elif isinstance(objs, (models.Column, models.Row)):
        objs = [_get_figures_core(child) for child in objs.children
                if not isinstance(child, (models.ToolbarBox,
                                          models.WidgetBox))]
    return objs

def _get_figures(objs):
    try:
        return list(flatten(_get_figures_core(objs)))
    except TypeError:
        return [_get_figures_core(objs)]

def _save_to_svg(hv_obj, save):
    bokeh_obj = hv.renderer('bokeh').get_plot(hv_obj).state
    figures = _get_figures(bokeh_obj)
    for i, figure in enumerate(figures):
        figure.output_backend = 'svg'

        if len(figures) != 1:
            if not os.path.exists(save):
                os.makedirs(save)
            tidied_title = figure.title.text
            save_fp = os.path.join(
                save, '{0}_{1}'.format(tidied_title, i))
        else:
            save_fp = save

        if not save_fp.endswith('svg'):
            save_fp = '{0}.{1}'.format(save_fp, 'svg')

        export_svgs(figure, save_fp)