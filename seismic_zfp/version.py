class SeismicZfpVersion:
    def __init__(self, arg):
        if isinstance(arg, str):
            version_numbers_tuple = tuple(part for part in arg.split("."))
            self.major = int(version_numbers_tuple[0])
            self.minor = int(version_numbers_tuple[1])
            self.patch = int(version_numbers_tuple[2])
            self.changes_exist = len(version_numbers_tuple) > 3
        elif isinstance(arg, int):
            self.major = arg//(1024*2048)
            self.minor = (arg - self.major*1024*2048) // 2048
            self.patch = (arg - self.major*1024*2048 - self.minor*2048) // 2
            self.changes_exist = arg % 2 == 0
        elif isinstance(arg, tuple):
            self.major = arg[0]
            self.minor = arg[1]
            self.patch = arg[2]
            self.changes_exist = len(arg) > 3

        self.encoding = self.to_encoding()
        self.string_version = self.to_string()

    def to_encoding(self):
        encoding = 1024*2048*self.major + 2048*self.minor + 2*self.patch + 1
        if self.changes_exist:
            encoding -= 1
        return encoding

    def to_string(self):
        string_version = ".".join([str(self.major), str(self.minor), str(self.patch)])
        if self.changes_exist:
            string_version += ".dev"
        return string_version

    def to_tuple(self):
        if self.changes_exist:
            return self.major, self.minor, self.patch, ".dev"
        else:
            return self.major, self.minor, self.patch

    def __gt__(self, other):
        return self.encoding > other.encoding

    def __repr__(self):
        return 'Version({})'.format(self.string_version)
