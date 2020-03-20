from __future__ import print_function
import segyio
import sys


def get_headerword_code(hw):
    return segyio.tracefield.keys[str(segyio.tracefield.TraceField(hw))]


def get_first_last_headers(segyfile):
    return segyfile.header[0].items(), segyfile.header[-1].items()


def get_nonzero_headerwords(segyfile):
        return [k for k, v in segyfile.header[0].items() if v != 0]


def get_invariant_headerwords(segyfile):
    first_header, last_header = get_first_last_headers(segyfile)
    return [k1 for (k1, v1), (kl, vl) in zip(first_header, last_header) if v1 == vl]


def get_variant_headerwords(segyfile):
    first_header, last_header = get_first_last_headers(segyfile)
    return [k1 for (k1, v1), (kl, vl) in zip(first_header, last_header) if v1 != vl]


def get_invariant_nonzero_headerwords(segyfile):
    return [header_word for header_word in get_nonzero_headerwords(segyfile)
            if header_word in get_invariant_headerwords(segyfile)]


def find_duplicated_headerwords(segyfile):
    variant_nonzero_header_words = get_variant_headerwords(segyfile)
    first_header, last_header = segyfile.header[0], segyfile.header[-1]
    hw_mappings = {}

    for i, hw in enumerate(variant_nonzero_header_words):
        for hw2 in variant_nonzero_header_words[:i]:
            if first_header[hw] == first_header[hw2] and last_header[hw] == last_header[hw2]:
                hw_mappings[hw] = hw2
                # Use the first one you find...
                break
    return hw_mappings


def get_unique_headerwords(segyfile):
    variant_header_words = get_variant_headerwords(segyfile)
    duplicate_header_words = find_duplicated_headerwords(segyfile)
    for i, hw in enumerate(variant_header_words):
        if hw in duplicate_header_words.keys():
            variant_header_words[i] = duplicate_header_words[hw]

    variant_header_words = list(set(variant_header_words))
    variant_header_word_codes = [get_headerword_code(header_words) for header_words in variant_header_words]

    return [x for _, x in sorted(zip(variant_header_word_codes, variant_header_words))]


def get_headerword_infolist(segyfile):
    """Determines which header fields have constant values, what those constant values are and
    which non-constant fields are duplicates of others. Achieves this by reading first and last
    traces of the provided segy file. Returns encoding of this information.

    Parameters
    ----------
    segyfile: str

    Returns
    -------
    hw_info_list: list of 3-tuples of integers:
    (Trace header start-byte, constant value, duplicated trace header start-byte)
    """
    invariant_nonzero_header_words = get_invariant_nonzero_headerwords(segyfile)
    unique_variant_nonzero_header_words = get_unique_headerwords(segyfile)
    duplicate_header_words = find_duplicated_headerwords(segyfile)

    hw_info_list = []
    for hw in segyfile.header[0]:
        if hw in invariant_nonzero_header_words:
            default = segyfile.header[0][hw]
        else:
            default = 0

        if hw in unique_variant_nonzero_header_words:
            mapping = get_headerword_code(hw)
        elif hw in duplicate_header_words.keys():
            mapping = get_headerword_code(duplicate_header_words[hw])
        else:
            mapping = 0

        hw_info_list.append((get_headerword_code(hw), default, mapping))

    return hw_info_list


if __name__ == '__main__':
    filename = sys.argv[1]

    with segyio.open(filename) as segyfile:
        hw_info_list = get_headerword_infolist(segyfile)
        print(hw_info_list)
