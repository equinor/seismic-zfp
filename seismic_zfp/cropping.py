import struct
import numpy as np

from .read import SgzReader
from .utils import pad, int_to_bytes, np_float_to_bytes, np_float_to_bytes_signed, coord_to_index
from .sgzconstants import DISK_BLOCK_BYTES, SEGY_TEXT_HEADER_BYTES

class SgzCropper(SgzReader):
    """Creates SGZ files from subcrops of others.
       This is superior to using read_subvolume() and a NumpyConverter because seismic data is copied compressed.
       However, it comes with the limitation that the indexes to crop on must align with the compression blocks"""

    def __init__(self, file, filetype_checking=True, preload=False, chunk_cache_size=None):
        super().__init__(file, filetype_checking, preload, chunk_cache_size)

    def check_and_correct_bounds(self, iline_index_range, xline_index_range, zslices_index_range):
        valid_bounds = True
        if iline_index_range is None and xline_index_range is None and zslices_index_range is None:
            print("Error: No cropping ranges specified, no file will be written.")
            valid_bounds = False

        if iline_index_range   is None: iline_index_range   = (0, len(self.ilines))
        if xline_index_range   is None: xline_index_range   = (0, len(self.xlines))
        if zslices_index_range is None: zslices_index_range = (0, len(self.zslices))

        err_string = "{} bounds out of range. Expected range within [{},{}], but got ({},{})."

        if iline_index_range[0] < 0 or iline_index_range[1] > len(self.ilines):
            print(err_string.format("Inline", 0, len(self.ilines), *iline_index_range))
            valid_bounds = False

        if xline_index_range[0] < 0 or xline_index_range[1] > len(self.xlines):
            print(err_string.format("Crossline", 0, len(self.xlines), *xline_index_range))
            valid_bounds = False

        if zslices_index_range[0] < 0 or zslices_index_range[1] > len(self.zslices):
            print(err_string.format("Zslice", 0, len(self.zslices), *zslices_index_range))
            valid_bounds = False

        if valid_bounds:
            iline_index_range = self.correct_bounds(iline_index_range, "inline", len(self.ilines), self.blockshape[0])
            xline_index_range = self.correct_bounds(xline_index_range, "crossline", len(self.xlines), self.blockshape[1])
            zslices_index_range = self.correct_bounds(zslices_index_range, "zslice", len(self.zslices), self.blockshape[2])
        else:
            raise IndexError("There is a chasm, Of carbon and silicon, The server can't bridge.")

        return iline_index_range, xline_index_range, zslices_index_range

    def correct_bounds(self, index_range, axis_name, axis_len, pad_dim):
        new_index_0 = index_range[0] - index_range[0] % pad_dim if index_range[0] % pad_dim != 0 else index_range[0]
        new_index_1 = index_range[1] - index_range[1] % pad_dim + pad_dim if index_range[1] % pad_dim != 0 else index_range[1]

        new_index_0 = max(new_index_0, 0)
        new_index_1 = min(new_index_1, axis_len)
        if not (new_index_0==index_range[0] and new_index_1==index_range[1]):
            print("Warning: Specified {} bounds not aligned with compression blocks.".format(axis_name))
            print("... correcting from ({},{}) to ({},{})".format(*index_range, new_index_0, new_index_1))
        return new_index_0, new_index_1

    def regenerate_header(self, iline_index_range, xline_index_range, zslices_index_range):
        len_zslices = zslices_index_range[1] - zslices_index_range[0]
        len_xlines = xline_index_range[1] - xline_index_range[0]
        len_ilines = iline_index_range[1] - iline_index_range[0]
        compressed_data_length_diskblocks = int(((self.rate * pad(len_zslices, self.blockshape[2]) *
                                                              pad(len_xlines, self.blockshape[1]) *
                                                              pad(len_ilines, self.blockshape[0]) ) // 8)
                                                // DISK_BLOCK_BYTES)

        header = bytearray(self.headerbytes).copy()
        header[4:8] = int_to_bytes(len_zslices)
        header[8:12] = int_to_bytes(len_xlines)
        header[12:16] = int_to_bytes(len_ilines)
        header[16:20] = np_float_to_bytes_signed(np.int32(self.zslices[zslices_index_range[0]]))
        header[20:24] = np_float_to_bytes(np.int32(self.xlines[xline_index_range[0]]))
        header[24:28] = np_float_to_bytes(np.int32(self.ilines[iline_index_range[0]]))
        header[56:60] = int_to_bytes(compressed_data_length_diskblocks)
        header[60:64] = int_to_bytes((len_xlines * len_ilines * 32) // 8)

        # We need to inform the SEG-Y binary header what has happened to the trace length, otherwise
        # segyio will get all confused if attempting to read the cropped SGZ converted back to SEG-Y
        header[DISK_BLOCK_BYTES + SEGY_TEXT_HEADER_BYTES + 20:
               DISK_BLOCK_BYTES + SEGY_TEXT_HEADER_BYTES + 22] = struct.pack('>H', len_zslices)

        return header

    @staticmethod
    def get_index_range(coord_range, coord_list):
        if coord_range is None:
            return None
        return (coord_to_index(coord_range[0], coord_list, include_stop=True),
                coord_to_index(coord_range[1], coord_list, include_stop=True))

    def write_cropped_file_by_coords(self, out_file, iline_coord_range=None,
                                                     xline_coord_range=None,
                                                     zslices_coord_range=None):
        """Convenience function for cropping SGZ files by coordinates rather than indexes

        Parameters
        ----------

        out_file: str
            Filepath to write cropped file to

        iline_coord_range: 2-tuple of ints
            The start:stop inline numbers for cropping

        xline_coord_range: 2-tuple of ints
            The start:stop crossline numbers for cropping

        zslices_coord_range: 2-tuple of ints
            The start:stop time/depth coordinates for cropping

        Raises
        ------

        IndexError
            If indexes to crop on are missing, or do not align with the compression blocks
        """
        self.write_cropped_file_by_indexes(out_file, self.get_index_range(iline_coord_range, self.ilines),
                                                     self.get_index_range(xline_coord_range, self.xlines),
                                                     self.get_index_range(zslices_coord_range, self.zslices))

    def write_cropped_file_by_indexes(self, out_file, iline_index_range=None,
                                                      xline_index_range=None,
                                                      zslices_index_range=None):
        """General entrypoint for cropping SGZ files

        Parameters
        ----------

        out_file: str
            Filepath to write cropped file to

        iline_index_range: 2-tuple of ints
            The start:stop indexes for cropping inlines to

        xline_index_range: 2-tuple of ints
            The start:stop indexes for cropping crosslines to

        zslices_index_range: 2-tuple of ints
            The start:stop indexes for cropping samples to

        Raises
        ------

        IndexError
            If indexes to crop on are missing, or do not align with the compression blocks
        """

        iline_index_range, xline_index_range, zslices_index_range = self.check_and_correct_bounds(iline_index_range,
                                                                                                  xline_index_range,
                                                                                                  zslices_index_range)

        z_units = (pad(zslices_index_range[1], self.blockshape[2]) - zslices_index_range[0]) // 4
        xl_units = (xline_index_range[1] - xline_index_range[0]) // 4
        il_units = (iline_index_range[1] - iline_index_range[0]) // 4

        header = self.regenerate_header(iline_index_range, xline_index_range, zslices_index_range)
        compressed_bytes = self.loader.read_chunk_range(iline_index_range[0],
                                                        xline_index_range[0],
                                                        zslices_index_range[0],
                                                        il_units, xl_units, z_units)
        with open(out_file, 'wb') as new_sgz_file:
            new_sgz_file.write(header)
            new_sgz_file.write(compressed_bytes)

            self.read_variant_headers()
            for k in self.stored_header_keys:
                header_array = self.variant_headers[k].reshape((self.n_ilines, self.n_xlines)).astype(np.int32)
                cropped_header_array = header_array[iline_index_range[0]:iline_index_range[1],
                                                    xline_index_range[0]:xline_index_range[1]]
                new_sgz_file.write(cropped_header_array.flatten().tobytes())
