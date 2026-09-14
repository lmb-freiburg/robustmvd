_CRITICAL = 50
_ERROR = 40
_WARNING = 30
_INFO = 20
_DEBUG = 10
_SPAM = 5
_NOTSET = 0

_levelToName = {
    _CRITICAL: 'CRITICAL',
    _ERROR: 'ERROR',
    _WARNING: 'WARNING',
    _INFO: 'INFO',
    _DEBUG: 'DEBUG',
    _SPAM: 'SPAM',
    _NOTSET: 'NOTSET',
}
_nameToLevel = {
    'CRITICAL': _CRITICAL,
    'ERROR': _ERROR,
    'WARNING': _WARNING,
    'INFO': _INFO,
    'DEBUG': _DEBUG,
    'SPAM': _SPAM,
    'NOTSET': _NOTSET,
}

_log_level = _nameToLevel['INFO']
_log_file_paths = []
_log_files = []


def add_log_file(log_file_path, flush_line=False):
    global _log_files
    global _log_file_paths

    if log_file_path not in _log_file_paths:
        _log_file_paths.append(log_file_path)
        if flush_line:
            _log_files.append(open(log_file_path, 'a', buffering=1))
        else:
            _log_files.append(open(log_file_path, 'a'))


def remove_log_file(log_file_path):
    global _log_files
    global _log_file_paths

    if log_file_path in _log_file_paths:
        index = _log_file_paths.index(log_file_path)
        _log_files[index].close()
        del _log_files[index]
        del _log_file_paths[index]


def clear_log_files():
    global _log_files
    global _log_file_paths
    for log_file in _log_files:
        log_file.close()
    _log_files = []
    _log_file_paths = []


def check_level(level_str):
    return _nameToLevel[level_str.upper()] >= _log_level


def set_log_level(level_str):
    global _log_level
    _log_level = _nameToLevel[level_str.upper()]
    print("Using log level: %s." % _levelToName[_log_level])


def _log(level, msg=None):
    if msg is not None:
        print(str(msg))
        for log_file in _log_files:
            log_file.write(str(msg) + "\n")
    else:
        print()
        for log_file in _log_files:
            log_file.write("\n")


def spam(msg=None):
    if _log_level > _nameToLevel['SPAM']:
        return
    else:
        _log("SPAM", msg)


def debug(msg=None):
    if _log_level > _nameToLevel['DEBUG']:
        return
    else:
        _log("DEBUG", msg)


def info(msg=None):
    if _log_level > _nameToLevel['INFO']:
        return
    else:
        _log("INFO", msg)


def warning(msg=None):
    if _log_level > _nameToLevel['WARNING']:
        return
    else:
        _log("WARNING", msg)


def error(msg=None):
    if _log_level > _nameToLevel['ERROR']:
        return
    else:
        _log("ERROR", msg)


def critical(msg=None):
    if _log_level > _nameToLevel['CRITICAL']:
        return
    else:
        _log("CRITICAL", msg)
