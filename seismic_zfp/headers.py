import segyio
import sys

#print(segyio.tracefield.TraceField(1))
#print(segyio.tracefield.keys["INLINE_3D"])


def getFirstLastHeaders(segyfile):
    return segyfile.header[0].items(), segyfile.header[-1].items()


def getNonZeroHeaderWordsFromSegyfile(segyfile):
        return [k for k, v in segyfile.header[0].items() if v != 0]


def getInvariantHeaderWordsFromSegyfile(segyfile):
    first_header, last_header = getFirstLastHeaders(segyfile)
    return [k1 for (k1, v1), (kl, vl) in zip(first_header, last_header) if v1 == vl]


def getVariantHeaderWordsFromSegyfile(segyfile):
    first_header, last_header = getFirstLastHeaders(segyfile)
    return [k1 for (k1, v1), (kl, vl) in zip(first_header, last_header) if v1 != vl]


def getInvariantNonzeroHeaderWordsFromSegyfile(segyfile):
    return [header_word for header_word in getNonZeroHeaderWordsFromSegyfile(segyfile)
            if header_word in getInvariantHeaderWordsFromSegyfile(segyfile)]


def findDuplicateHeaderWordsFromSegyFile(segyfile):
    variant_nonzero_header_words = getVariantHeaderWordsFromSegyfile(segyfile)
    first_header, last_header = segyfile.header[0], segyfile.header[-1]
    hw_mappings = {}

    for i, hw in enumerate(variant_nonzero_header_words):
        for hw2 in variant_nonzero_header_words[:i]:
            if first_header[hw] == first_header[hw2] and last_header[hw] == last_header[hw2]:
                hw_mappings[hw] = hw2
                # Use the first one you find...
                break
    return hw_mappings


def findUniqueHeaderWordsFromSegyFile(segyfile):
    variant_header_words = getVariantHeaderWordsFromSegyfile(segyfile)
    duplicate_header_words = findDuplicateHeaderWordsFromSegyFile(segyfile)
    for i, hw in enumerate(variant_header_words):
        if hw in duplicate_header_words.keys():
            variant_header_words[i] = duplicate_header_words[hw]

    return set(variant_header_words)


def getHeaderwordInfoList(segyfile):
    invariant_nonzero_header_words = getInvariantNonzeroHeaderWordsFromSegyfile(segyfile)
    unique_variant_nonzero_header_words = findUniqueHeaderWordsFromSegyFile(segyfile)
    duplicate_header_words = findDuplicateHeaderWordsFromSegyFile(segyfile)

    hw_info_list = []
    for hw in segyfile.header[0]:
        if hw in invariant_nonzero_header_words:
            default = segyfile.header[0][hw]
        else:
            default = 0

        if hw in unique_variant_nonzero_header_words:
            mapping = hw
        elif hw in duplicate_header_words.keys():
            mapping = duplicate_header_words[hw]
        else:
            mapping = None

        hw_info_list.append((hw, default, mapping))
    return hw_info_list


if __name__ == '__main__':
    filename = sys.argv[1]

    with segyio.open(filename) as segyfile:
        hw_info_list = getHeaderwordInfoList(segyfile)
        print(hw_info_list)
