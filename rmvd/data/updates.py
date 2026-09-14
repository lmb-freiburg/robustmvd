import os.path as osp
import abc
import pickle

from rmvd.utils import logging


class Update(metaclass=abc.ABCMeta):    
    def __init__(self):
        pass

    @abc.abstractmethod
    def load(self, orig_sample_dict, root=None):
        return


class Updates:
    def __init__(self, name, updates=None, root=None, prefix=None, postfix=None, verbose=True):
        self.name = name
        self.verbose = verbose
        self.path = None  # path of the file that the updates were created from; set in from_file

        if self.verbose:
            logging.info(f"Initializing dataset updates {self.name}.")

        self.root = root
        self.prefix = prefix
        self.postfix = postfix

        self.updates = {} if updates is None else updates

        if self.verbose:
            if self.root is not None:
                logging.info(f"\tData root: {self.root}")
            logging.info(f"\tPrefix: {self.prefix}")
            logging.info(f"\tPostfix: {self.postfix}")
            logging.info(f"\tNumber of updates: {len(self)}")
            logging.info(f"Finished initializing dataset update {self.name}.")

    @classmethod
    def from_file(cls, path, **kwargs):
        name = osp.splitext(osp.split(path)[1])[0]

        with open(path, "rb") as file:
            updates = pickle.load(file)

        update = cls(name=name, updates=updates, **kwargs)
        update.path = path
        return update

    def write(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.updates, file)

    def apply_update(self, sample_dict, index):
        if index in self:
            update = self[index]
            update_dict = update.load(orig_sample_dict=sample_dict, root=self.root)
            update_dict = self._add_pre_post_fixes(update_dict)
            sample_dict.update(update_dict)

    def _add_pre_post_fixes(self, update_dict):
        update_dict_out = {}

        for key, val in update_dict.items():
            key = (self.prefix if self.prefix is not None else "") + key + (self.postfix if self.postfix is not None else "")
            update_dict_out[key] = val

        return update_dict_out

    def __getitem__(self, index):
        return self.updates[index]

    def __contains__(self, index):
        return index in self.updates

    def __len__(self):
        return len(self.updates)
