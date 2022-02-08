import collections
import numpy as np
import segyio

from .utils import signed_int_to_bytes, bytes_to_signed_int, FileOffset
from .seismicfile import Filetype
from .sgzconstants import DISK_BLOCK_BYTES

class HeaderwordInfo:
    """Represents the variant/invariant/unpopulated status of trace headers, and their values.

    Three mutually exclusive modes of operation:

    - seismicfile:   Determines which header fields have constant values, what those constant values are and
    which non-constant fields are duplicates of others. Achieves this by reading first and last
    traces of the provided segy file. Returns encoding of this information.

    - variant_header_list:   List of header fields which are to be treated as variant

    - variant_header_dict:   Dict of variant header fields for creating new file
    """
    def __init__(self, n_traces, seismicfile=None,
                                 variant_header_list=None,
                                 variant_header_dict=None,
                                 header_detection=None,
                                 buffer=None):
        """
        Parameters
        ----------
        n_traces: Number of traces in output file

        seismicfile: segyio filehandle (optional - pick one)

        variant_header_list: list of headerwords which need storing (optional - pick one)

        variant_header_dict: dictionary of {headerword: numpy array, } (optional - pick one)
        """
        # Check only one of the options for creating a HeaderwordInfo class is used
        if sum([_ is not None for _ in [seismicfile, variant_header_list, variant_header_dict, buffer]]) != 1:
            raise RuntimeError("Must specify at least one of seismicfile and variant_header_list for constructor")

        self.header_detection = header_detection
        self.table = {self._get_hw_code(hw): (0, 0) for hw in segyio.segy.Field(bytearray(240), kind='trace')}

        if seismicfile is not None:
            if seismicfile.filetype in [Filetype.SEGY, Filetype.VDS]:
                # This would work for ZGY too, but there's a more efficient way to do it
                self.seismicfile = seismicfile
                self.unique_variant_nonzero_header_words = self._get_unique_headerwords()
                self.duplicate_header_words = self._find_duplicated_headerwords()

                for hw in seismicfile.header[0]:
                    if hw in self._get_invariant_nonzero_headerwords():
                        self.table[self._get_hw_code(hw)] = (seismicfile.header[0][hw], 0)

                    if hw in self.unique_variant_nonzero_header_words:
                        self.table[self._get_hw_code(hw)] = (0, self._get_hw_code(hw))
                    elif hw in self.duplicate_header_words.keys():
                        self.table[self._get_hw_code(hw)] = (0, self._get_hw_code(self.duplicate_header_words[hw]))

                self.headers_dict = collections.OrderedDict.fromkeys(self.unique_variant_nonzero_header_words)
                for k in self.unique_variant_nonzero_header_words:
                    self.headers_dict[k] = np.zeros(n_traces, dtype=np.int32)

            elif seismicfile.filetype == Filetype.ZGY:
                # Set up hw table with entries possible to scrape out of ZGY
                self.table[115] = (int(seismicfile.n_samples), 0)
                self.table[117] = (int(1000*seismicfile.zinc), 0)
                for hw_code in [181, 185, 189, 193]:
                    self.table[hw_code] = (0, hw_code)

                # Create required numpy int arrays
                cdp_x, cdp_y, iline_headers, xline_headers = self.get_zgy_header_arrays(seismicfile)

                # Keep them in memory until file is ready for them to be written
                self.headers_dict = {181: cdp_x, 185: cdp_y, 189: iline_headers, 193:xline_headers}

            else:
                raise RuntimeError("Only SEG-Y and ZGY files supported for header generation")

        elif variant_header_list is not None:
            self.unique_variant_nonzero_header_words = variant_header_list
            for hw in variant_header_list:
                self.table[self._get_hw_code(hw)] = (0, self._get_hw_code(hw))

            self.headers_dict = collections.OrderedDict.fromkeys(self.unique_variant_nonzero_header_words)
            for k in self.unique_variant_nonzero_header_words:
                self.headers_dict[k] = np.zeros(n_traces, dtype=np.int32)

        elif variant_header_dict is not None:
            self.unique_variant_nonzero_header_words = variant_header_dict.keys()
            for hw in variant_header_dict.keys():
                self.table[self._get_hw_code(hw)] = (0, self._get_hw_code(hw))
            self.headers_dict = variant_header_dict

        elif buffer is not None:
            template = [tuple((bytes_to_signed_int(buffer[i * 12 + j:i * 12 + j + 4])
                               for j in range(0, 12, 4))) for i in range(89)]
            for hv in template:
                self.table[hv[0]] = (hv[1], hv[2])


    def get_header_dict(self, n_header_arrays, n_header_blocks, compressed_data_diskblocks, padded_header_entry_length_bytes):
        header_dict = {}
        stored_header_keys = []

        for k, v in self.table.items():
            tf = segyio.tracefield.TraceField(k)
            if v[0] != 0 or v[1] == 0:
                # In these cases we have an invariant value
                header_dict[tf] = v[0]

            elif segyio.tracefield.TraceField(v[1]) in header_dict.keys():
                # We have a previously discovered header value
                header_dict[tf] = header_dict[segyio.tracefield.TraceField(v[1])]
            else:
                # This is a new header value
                header_dict[tf] = FileOffset(DISK_BLOCK_BYTES * n_header_blocks +
                                             DISK_BLOCK_BYTES * compressed_data_diskblocks +
                                             len(stored_header_keys) * padded_header_entry_length_bytes)
                stored_header_keys.append(tf)

        # We should find the same number of headers arrays as have been written!
        assert(len(stored_header_keys) == n_header_arrays)

        return header_dict


    def get_zgy_header_arrays(self, seismicfile):
        iline_axis = np.linspace(seismicfile.ilines[0], seismicfile.ilines[-1],
                                 num=len(seismicfile.ilines), dtype=np.intc)
        xline_axis = np.linspace(seismicfile.xlines[0], seismicfile.xlines[-1],
                                 num=len(seismicfile.xlines), dtype=np.intc)
        xline_headers, iline_headers = np.meshgrid(xline_axis, iline_axis)

        iline_axis = np.linspace(0, seismicfile.n_ilines - 1, num=seismicfile.n_ilines)
        xline_axis = np.linspace(0, seismicfile.n_xlines - 1, num=seismicfile.n_xlines)
        xline_idx, iline_idx = np.meshgrid(xline_axis, iline_axis)

        corners = seismicfile.corners

        easting_inc_il = (corners[1][0] - corners[0][0]) / (seismicfile.n_ilines - 1)
        northing_inc_il = (corners[1][1] - corners[0][1]) / (seismicfile.n_ilines - 1)
        easting_inc_xl = (corners[2][0] - corners[0][0]) / (seismicfile.n_xlines - 1)
        northing_inc_xl = (corners[2][1] - corners[0][1]) / (seismicfile.n_xlines - 1)

        cdp_x = np.round_(100.0 * (corners[0][0] + iline_idx * easting_inc_il + xline_idx * easting_inc_xl)).astype(np.intc)
        cdp_y = np.round_(100.0 * (corners[0][1] + iline_idx * northing_inc_il + xline_idx * northing_inc_xl)).astype(np.intc)

        return cdp_x, cdp_y, iline_headers, xline_headers

    def __repr__(self):
        output = ""
        for row in self.to_list():
            output += "{} | {} | {}\n".format(row[0], row[1], row[2])
        return output

    def update_table(self, key, new_value):
        self.table[key] = new_value

    def to_list(self):
        """
        Returns
        -------
        hw_info_list: list of 3-tuples of integers:
        (Trace header start-byte, constant value, duplicated trace header start-byte)
        """
        return [(key, value[0], value[1]) for key, value in self.table.items()]

    def to_buffer(self):
        buf = bytearray(1068)
        for i, hw_info in enumerate(self.to_list()):
            start = i * 12
            buf[start + 0:start + 4] = signed_int_to_bytes(hw_info[0])
            buf[start + 4:start + 8] = signed_int_to_bytes(hw_info[1])
            buf[start + 8:start + 12] = signed_int_to_bytes(hw_info[2])
        return buf

    def get_header_array_count(self):
        return sum(hw[0] == hw[2] for hw in self.to_list())

    @staticmethod
    def _get_hw_code(hw):
        return segyio.tracefield.keys[str(segyio.tracefield.TraceField(hw))]

    def _get_first_last_headers(self):
        return self.seismicfile.header[0].items(), self.seismicfile.header[-1].items()

    def _get_nonzero_headerwords(self):
        return [k for k, v in self.seismicfile.header[0].items() if v != 0]

    def _get_invariant_headerwords(self):
        first_header, last_header = self._get_first_last_headers()
        return [k1 for (k1, v1), (kl, vl) in zip(first_header, last_header) if v1 == vl]

    def _get_variant_headerwords(self):
        first_header, last_header = self._get_first_last_headers()
        return [k1 for (k1, v1), (kl, vl) in zip(first_header, last_header) if v1 != vl]

    def _get_invariant_nonzero_headerwords(self):
        return [header_word for header_word in self._get_nonzero_headerwords()
                if header_word in self._get_invariant_headerwords()]

    def _find_duplicated_headerwords(self):
        variant_nonzero_header_words = self._get_variant_headerwords()
        first_header, last_header = self.seismicfile.header[0], self.seismicfile.header[-1]
        hw_mappings = {}

        for i, hw in enumerate(variant_nonzero_header_words):
            for hw2 in variant_nonzero_header_words[:i]:
                if first_header[hw] == first_header[hw2] and last_header[hw] == last_header[hw2]:
                    hw_mappings[hw] = hw2
                    break # Use the first one you find...

        return hw_mappings

    def _get_unique_headerwords(self):
        variant_header_words = self._get_variant_headerwords()
        duplicate_header_words = self._find_duplicated_headerwords()
        for i, hw in enumerate(variant_header_words):
            if hw in duplicate_header_words.keys():
                variant_header_words[i] = duplicate_header_words[hw]

        variant_header_words = list(set(variant_header_words))
        variant_header_word_codes = [self._get_hw_code(header_words) for header_words in variant_header_words]

        return [x for _, x in sorted(zip(variant_header_word_codes, variant_header_words))]
