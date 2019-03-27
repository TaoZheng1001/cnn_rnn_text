#_*_ coding:utf-8 _*_

import tensorflow as tf
import os
import sys
import numpy as np
import math
from Vocab import Vocab


class CategoryDict:
    def __init__(self,filename):
        self._category_to_id = {}
        with open(filename,'rb') as f:
            lines = f.readlines()
        for line in lines:
            line = line.decode()
            category = line.strip('\r\n')
            idx = len(self._category_to_id)
            self._category_to_id[category] = idx

    def size(self):
        return len(self._category_to_id)

    def category_to_id(self, category):
        if not category in self._category_to_id:
            raise Execption(
                "%s is not in our category list" % category)
        return self._category_to_id[category]


