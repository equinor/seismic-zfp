from collections.abc import Mapping
from .read import SgzReader
from .utils import coord_to_index


class SubvolumeAccessor(SgzReader):

    def __init__(self, file):
        super(SubvolumeAccessor, self).__init__(file)

        self.zslices_int = self.zslices.astype('intc')

        self.axes_message = (f"Inline {self._axis_message(self.ilines)}, "
                             f"Crossline {self._axis_message(self.xlines)}, "
                             f"Samples {self._axis_message(self.zslices_int)}")

    @staticmethod
    def _axis_message(coords):
        """start:stop:step of an axis, with stop exclusive as in the subscripts"""
        step = coords[1] - coords[0]
        return f"{coords[0]}:{coords[-1] + step}:{step}"

    def __getitem__(self, subscripts):
        il, xl, z = subscripts

        self._check_subscripts(il, self.ilines, "Inline")
        self._check_subscripts(xl, self.xlines, "Crossline")
        self._check_subscripts(z,  self.zslices_int, "Samples")

        il_start, il_step, il_stop = self._get_index_subscripts(il, self.ilines)
        xl_start, xl_step, xl_stop = self._get_index_subscripts(xl, self.xlines)
        z_start, z_step, z_stop = self._get_index_subscripts(z, self.zslices_int)

        # N.B. While this implementation will work with steps larger than 1, it has to read and decompress
        # everything in between. Of course there's no way to avoid that for steps <=4 but memory usage could
        # be reduced with a better implementation
        return self.read_subvolume(il_start, il_stop,
                                   xl_start, xl_stop,
                                   z_start, z_stop)[::il_step, ::xl_step, ::z_step]

    @staticmethod
    def _get_index_subscripts(coord_subscript, coords):
        start = 0 if coord_subscript.start is None else coord_to_index(coord_subscript.start, coords)
        if (coord_subscript.stop is None) or (coord_subscript.stop == coords[-1] + coords[1] - coords[0]):
            stop = len(coords)
        else:
            stop = coord_to_index(coord_subscript.stop, coords)
        step = 1 if coord_subscript.step is None else coord_subscript.step // (coords[1] - coords[0])
        return start, step, stop

    def _check_subscripts(self, subscript, coords, coord_name):
        if subscript.start is not None and not coords[0] <= subscript.start < coords[-1] + coords[1] - coords[0]:
            raise IndexError(f"{coord_name} start {subscript.start} out of range. Axes are {self.axes_message}")
        if subscript.stop is not None and not coords[0] < subscript.stop <= coords[-1] + coords[1] - coords[0]:
            raise IndexError(f"{coord_name} stop {subscript.stop} out of range. Axes are {self.axes_message}")
        if subscript.step is not None and not subscript.step % (coords[1] - coords[0]) == 0:
            raise IndexError(f"{coord_name} step {subscript.step} invalid. Axes are {self.axes_message}")


class SubvolumeAccessor4d(SubvolumeAccessor):
    """subvolume[il, xl, offset, z] for prestack files, all four subscripts being slices in coordinate
    units (line numbers, offset values, sample times), as for the 3D SubvolumeAccessor. Unlike gather,
    a range of samples may be requested, and the offsets may be restricted to a subset."""

    def __init__(self, file):
        super(SubvolumeAccessor, self).__init__(file)
        self.zslices_int = self.zslices.astype('intc')
        self.axes_message = (f"Inline {self._axis_message(self.ilines)}, "
                             f"Crossline {self._axis_message(self.xlines)}, "
                             f"Offset {self._axis_message(self.offsets)}, "
                             f"Samples {self._axis_message(self.zslices_int)}")

    def __getitem__(self, subscripts):
        if not isinstance(subscripts, tuple) or len(subscripts) != 4:
            raise TypeError("4D subvolume requires [iline, crossline, offset, sample] slices")
        axes = ((self.ilines, "Inline"), (self.xlines, "Crossline"),
                (self.offsets, "Offset"), (self.zslices_int, "Samples"))
        bounds = []
        for subscript, (coords, name) in zip(subscripts, axes):
            self._check_subscripts(subscript, coords, name)
            bounds.append(self._get_index_subscripts(subscript, coords))
        (il_start, il_step, il_stop), (xl_start, xl_step, xl_stop), \
            (off_start, off_step, off_stop), (z_start, z_step, z_stop) = bounds
        return self.read_subvolume_4d(il_start, il_stop, xl_start, xl_stop,
                                      off_start, off_stop, z_start, z_stop)[::il_step, ::xl_step, ::off_step, ::z_step]


class Accessor(SgzReader, Mapping):

    def __iter__(self):
        return iter(self[:])

    def __len__(self):
        return self.len_object

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            # Acquiris Quodcumquae Rapis
            start, stop, step = subscript.indices(len(self))
            return [self.values_function(index) for index in range(start, stop, step)]
        elif subscript < 0:
            return self.values_function(len(self)+subscript)
        else:
            return self.values_function(subscript)


class SliceAccessor(Accessor):
    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            # Acquiris Quodcumquae Rapis
            start, stop, step = subscript.start, subscript.stop, subscript.step
            if step is None:
                step = int(self.keys_object[1] - self.keys_object[0])
            if start is None:
                start = int(self.keys_object[0])
            if stop is None:
                stop = int(self.keys_object[-1] + 1)
            return [self.values_function(index) for index in range(start, stop, step)]
        else:
            return self.values_function(subscript)


class InlineAccessor(SliceAccessor):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_ilines
        self.keys_object = self.ilines
        self.values_function = self.read_inline_number


class CrosslineAccessor(SliceAccessor):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_xlines
        self.keys_object = self.xlines
        self.values_function = self.read_crossline_number


class ZsliceAccessor(Accessor):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_samples
        self.keys_object = self.zslices
        self.values_function = self.read_zslice


class HeaderAccessor(Accessor):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.tracecount
        self.keys_object = range(self.tracecount)
        self.values_function = self.gen_trace_header


class TraceAccessor(Accessor):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.tracecount
        self.keys_object = range(self.tracecount)
        self.values_function = self.get_trace


class PrestackLineAccessor(SliceAccessor):
    """Emulates segyio's iline/xline for prestack files: [line] gives the first offset,
    [line, offset] a specific one, and slices on either give a generator of one array per
    combination, lines outermost and offsets innermost, as segyio does"""

    def _expand(self, subscript, coords):
        if isinstance(subscript, slice):
            start = int(coords[0]) if subscript.start is None else subscript.start
            stop = int(coords[-1] + 1) if subscript.stop is None else subscript.stop
            step = int(coords[1] - coords[0]) if subscript.step is None else subscript.step
            return list(range(start, stop, step))
        return [subscript]

    def __getitem__(self, subscript):
        line, offset = subscript if isinstance(subscript, tuple) else (subscript, self.offsets[0])
        if isinstance(line, slice) or isinstance(offset, slice):
            return (self.values_function(l, o) for l in self._expand(line, self.keys_object)
                    for o in self._expand(offset, self.offsets))
        return self.values_function(line, offset)


class InlineAccessor4d(PrestackLineAccessor):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_ilines
        self.keys_object = self.ilines

    def values_function(self, il_no, offset_no):
        il_id, offset_id = self.get_inline_index(il_no), self.get_offset_index(offset_no)
        return self.read_subvolume_4d(il_id, il_id + 1, 0, self.n_xlines,
                                      offset_id, offset_id + 1, 0, self.n_samples)[0, :, 0, :]


class CrosslineAccessor4d(PrestackLineAccessor):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_xlines
        self.keys_object = self.xlines

    def values_function(self, xl_no, offset_no):
        xl_id, offset_id = self.get_crossline_index(xl_no), self.get_offset_index(offset_no)
        return self.read_subvolume_4d(0, self.n_ilines, xl_id, xl_id + 1,
                                      offset_id, offset_id + 1, 0, self.n_samples)[:, 0, 0, :]


class ZsliceAccessor4d(Accessor):
    """segyio's depth_slice returns the first offset for prestack files"""
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_samples
        self.keys_object = self.zslices

    def values_function(self, zslice_id):
        if not 0 <= zslice_id < self.n_samples:
            raise IndexError(self.range_error.format(zslice_id, 0, self.n_samples - 1))
        return self.read_subvolume_4d(0, self.n_ilines, 0, self.n_xlines, 0, 1, zslice_id, zslice_id + 1)[:, :, 0, 0]


class GatherAccessor(PrestackLineAccessor):
    """Emulates segyio's gather: [il, xl] gives all offsets as (n_offsets, n_samples), [il, xl, offset]
    a single trace and [il, xl, offset_slice] the selected offsets. Slices on il/xl give a generator."""
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_ilines * self.n_xlines
        self.keys_object = self.ilines

    def __getitem__(self, subscript):
        if not isinstance(subscript, tuple) or not 2 <= len(subscript) <= 3:
            raise TypeError("gather requires [iline, crossline] or [iline, crossline, offset]")
        il, xl = subscript[0], subscript[1]
        offset = subscript[2] if len(subscript) == 3 else slice(None)
        if isinstance(il, slice) or isinstance(xl, slice):
            return (self.values_function(i, x, offset) for i in self._expand(il, self.ilines)
                    for x in self._expand(xl, self.xlines))
        return self.values_function(il, xl, offset)

    def values_function(self, il_no, xl_no, offset):
        il_id, xl_id = self.get_inline_index(il_no), self.get_crossline_index(xl_no)
        if isinstance(offset, slice):
            offset_ids = [self.get_offset_index(o) for o in self._expand(offset, self.offsets)]
            gather = self.read_subvolume_4d(il_id, il_id + 1, xl_id, xl_id + 1,
                                            offset_ids[0], offset_ids[-1] + 1, 0, self.n_samples)[0, 0]
            return gather[::offset_ids[1] - offset_ids[0]] if len(offset_ids) > 1 else gather
        offset_id = self.get_offset_index(offset)
        return self.read_subvolume_4d(il_id, il_id + 1, xl_id, xl_id + 1,
                                      offset_id, offset_id + 1, 0, self.n_samples)[0, 0, 0]
