#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

import six
import tensorflow as tf
from six.moves import range


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, num_of_group=0):
    """Compute the union of the current variables and checkpoint variables.
    albert/bert version"""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    init_vars = tf.train.list_variables(init_checkpoint)
    init_vars_name = [name for (name, _) in init_vars]

    if num_of_group > 0:
        assignment_map = []
        for gid in range(num_of_group):
            assignment_map.append(collections.OrderedDict())
    else:
        assignment_map = collections.OrderedDict()

    for name in name_to_variable:
        if name in init_vars_name:
            tvar_name = name
        elif (re.sub(r"/group_\d+/", "/group_0/",
                     six.ensure_str(name)) in init_vars_name and
              num_of_group > 1):
            tvar_name = re.sub(r"/group_\d+/", "/group_0/", six.ensure_str(name))
        elif (re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
              in init_vars_name and num_of_group > 1):
            tvar_name = re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
        elif (re.sub(r"/attention_\d+/", "/attention_1/", six.ensure_str(name))
              in init_vars_name and num_of_group > 1):
            tvar_name = re.sub(r"/attention_\d+/", "/attention_1/",
                               six.ensure_str(name))
        else:
            tf.logging.info("name %s does not get matched", name)
            continue
        tf.logging.info("name %s match to %s", name, tvar_name)
        if num_of_group > 0:
            group_matched = False
            for gid in range(1, num_of_group):
                if (("/group_" + str(gid) + "/" in name) or
                        ("/ffn_" + str(gid) + "/" in name) or
                        ("/attention_" + str(gid) + "/" in name)):
                    group_matched = True
                    tf.logging.info("%s belongs to %dth", name, gid)
                    assignment_map[gid][tvar_name] = name
            if not group_matched:
                assignment_map[0][tvar_name] = name
        else:
            assignment_map[tvar_name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[six.ensure_str(name) + ":0"] = 1

    return assignment_map, initialized_variable_names
