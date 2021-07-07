import numpy as np

from .read import SgzReader
from .utils import pad, int_to_bytes, np_float_to_bytes, np_float_to_bytes_signed
from .sgzconstants import DISK_BLOCK_BYTES


class SgzCropper(SgzReader):
    """Creates SGZ files from subcrops of others."""


    def __init__(self, file, filetype_checking=True, preload=False, chunk_cache_size=None):
        super().__init__(file, filetype_checking, preload, chunk_cache_size)

    def check_bounds(self, iline_range, xline_range, samples_range):
        valid_bounds = True
        if iline_range is None and xline_range is None and samples_range is None:
            print("Error: No cropping ranges specified, no file will be written.")
            valid_bounds = False

        if iline_range[0] < self.ilines[0] or iline_range[1] > self.ilines[-1]:
            print("Inline bounds out of range. ")

        if not valid_bounds:
            raise RuntimeError("There is a chasm, Of carbon and silicon, The server can't bridge.")


    def regenerate_header(self, iline_range, zslices_range, xline_range):
        len_zslices = int((zslices_range[1] - zslices_range[0]) // (self.zslices[1] - self.zslices[0]))
        len_xlines = (xline_range[1] - xline_range[0]) // (self.xlines[1] - self.xlines[0])
        len_ilines = (iline_range[1] - iline_range[0]) // (self.ilines[1] - self.ilines[0])
        compressed_data_length_diskblocks = int(((self.rate * pad(len_zslices, 512) * len_xlines * len_ilines) // 8)
                                                // DISK_BLOCK_BYTES)

        header = bytearray(self.headerbytes).copy()
        header[4:8] = int_to_bytes(len_zslices)
        header[8:12] = int_to_bytes(len_xlines)
        header[12:16] = int_to_bytes(len_ilines)
        header[16:20] = np_float_to_bytes_signed(np.int32(zslices_range[0]))
        header[20:24] = np_float_to_bytes(np.int32(xline_range[0]))
        header[24:28] = np_float_to_bytes(np.int32(iline_range[0]))
        header[56:60] = int_to_bytes(compressed_data_length_diskblocks)
        header[60:64] = int_to_bytes((len_xlines * len_ilines * 32) // 8)
        return header


    def write_cropped_file(self, out_file, iline_range=None, xline_range=None, zslices_range=None):
        """Specify iline_range, xline_range, zslices_range as tuples of start:stop ints"""
        self.check_bounds(iline_range, xline_range, zslices_range)

        header = self.regenerate_header(iline_range, zslices_range, xline_range)

        z_units = int((zslices_range[1] + 3*int(self.zslices[1]-self.zslices[0])) // 4 - zslices_range[0] // 4) // int(self.zslices[1]-self.zslices[0])
        xl_units = int((xline_range[1]-self.xlines[0] + 3) // 4 - (xline_range[0]-self.xlines[0]) // 4)
        il_units = int((iline_range[1]-self.ilines[0] + 3) // 4 - (iline_range[0]-self.ilines[0]) // 4)

        compressed_bytes = self.loader.read_chunk_range(int(iline_range[0]-self.ilines[0]),
                                                        int(xline_range[0]-self.xlines[0]),
                                                        int(zslices_range[0]-self.zslices[0]),
                                                        il_units, xl_units, 256) #256 is magic... calculate it properly!
        with open(out_file, 'wb') as new_sgz_file:
            new_sgz_file.write(header)
            new_sgz_file.write(compressed_bytes)

            self.read_variant_headers()
            for k in self.stored_header_keys:
                header_array = self.variant_headers[k].reshape((self.n_ilines, self.n_xlines)).astype(np.int32)
                cropped_header_array = header_array[iline_range[0]-self.ilines[0]:iline_range[1]-self.ilines[0],
                                                    xline_range[0]-self.xlines[0]:xline_range[1]-self.xlines[0]]
                new_sgz_file.write(cropped_header_array.flatten().tobytes())
