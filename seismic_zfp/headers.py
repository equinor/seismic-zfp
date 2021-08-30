import collections
import numpy as np
import segyio

from .utils import signed_int_to_bytes

class HeaderwordInfo:
    """Represents the variant/invariant/unpopulated status of trace headers, and their values.

    Three mutually exclusive modes of operation:

    - seismicfile:   Determines which header fields have constant values, what those constant values are and
    which non-constant fields are duplicates of others. Achieves this by reading first and last
    traces of the provided segy file. Returns encoding of this information.

    - variant_header_list:   List of header fields which are to be treated as variant

    - variant_header_dict:   Dict of variant header fields for creating new file
    """
    def __init__(self, n_traces, seismicfile=None, variant_header_list=None, variant_header_dict=None, header_detection=None):
        """
        Parameters
        ----------
        n_traces: Number of traces in output file

        seismicfile: segyio filehandle (optional - pick one)

        variant_header_list: list of headerwords which need storing (optional - pick one)

        variant_header_dict: dictionary of {headerword: numpy array, } (optional - pick one)
        """
        # Check only one of the options for creating a HeaderwordInfo class is used
        assert sum([_ is not None for _ in [seismicfile, variant_header_list, variant_header_dict]]) == 1
        self.header_detection = header_detection

        self.table = {}

        if seismicfile is not None:
            self.seismicfile = seismicfile
            self.unique_variant_nonzero_header_words = self._get_unique_headerwords()
            self.duplicate_header_words = self._find_duplicated_headerwords()

            for hw in seismicfile.header[0]:
                if hw in self._get_invariant_nonzero_headerwords():
                    default = seismicfile.header[0][hw]
                else:
                    default = 0

                if hw in self.unique_variant_nonzero_header_words:
                    mapping = self._get_headerword_code(hw)
                elif hw in self.duplicate_header_words.keys():
                    mapping = self._get_headerword_code(self.duplicate_header_words[hw])
                else:
                    mapping = 0
                self.table[self._get_headerword_code(hw)] = (default, mapping)

            self.headers_dict = collections.OrderedDict.fromkeys(self.unique_variant_nonzero_header_words)
            for k in self.unique_variant_nonzero_header_words:
                self.headers_dict[k] = np.zeros(n_traces, dtype=np.int32)


        elif variant_header_list is not None:
            self.unique_variant_nonzero_header_words = variant_header_list
            for hw in variant_header_list:
                self.table[self._get_headerword_code(hw)] = (0, self._get_headerword_code(hw))

            self.headers_dict = collections.OrderedDict.fromkeys(self.unique_variant_nonzero_header_words)
            for k in self.unique_variant_nonzero_header_words:
                self.headers_dict[k] = np.zeros(n_traces, dtype=np.int32)


        elif variant_header_dict is not None:
            self.unique_variant_nonzero_header_words = variant_header_dict.keys()
            for hw in variant_header_dict.keys():
                self.table[self._get_headerword_code(hw)] = (0, 0)

            self.headers_dict = variant_header_dict

        else:
            raise(RuntimeError, "Must specify at least one of seismicfile and variant_header_list for constructor")

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
    def _get_headerword_code(hw):
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
        variant_header_word_codes = [self._get_headerword_code(header_words) for header_words in variant_header_words]

        return [x for _, x in sorted(zip(variant_header_word_codes, variant_header_words))]
