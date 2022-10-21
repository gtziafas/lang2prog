import json
import numpy as np 

from programs.metadata.clevr import CLEVR_ATTRIBUTES
from structs import *


# converts raw CLEVR scene annotations to a list of
# object dictionaries, containing all relevant information
# and a unique identifier.
def load_clevr_scenes(scenes_json):
    with open(scenes_json) as f:
        scenes_dict = json.load(f)['scenes']
    scenes = []
    for s in scenes_dict:
        table = []
        for i, o in enumerate(s['objects']):
            item = {}
            item['id'] = '%d-%d' % (s['image_index'], i)
            if '3d_coords' in o:
                item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
                                    np.dot(o['3d_coords'], s['directions']['front']),
                                    o['3d_coords'][2]]
            else:
                item['position'] = o['position']
            item['color'] = o['color']
            item['material'] = o['material']
            item['shape'] = o['shape']
            item['size'] = o['size']
            table.append(item)
        scenes.append(table)
    return scenes


# change the order of the functions so next step takes always input of previous step
# except two-argument primitives, that will use a stack as first argument
# in CLEVR this can be fixed manually.
def clevr_convert_to_chain(programs):
    programs = programs.copy()
    query_fns = ['query_color', 'query_shape', 'query_size', 'query_material']
    equal_fns = ['equal_color', 'equal_shape', 'equal_size', 'equal_material']
    is_double_arg = lambda fn: fn.startswith('equal') or fn in ['union', 'intersect', 'greater_than', 'less_than']
    for i, p in enumerate(programs):
        funcs = [node['function'] for node in p]
        changed = False

        if funcs[0] == funcs[1] == 'scene': # weird but happens
            for node in p[1:]:
                if is_double_arg(node['function']):
                    _idx = node['inputs'][0] + 1
                    break
            new_program = p[1:_idx] + [p[0]] + p[_idx:]
            changed = True
            for j, node in enumerate(new_program):
                if node['function'] == 'scene':
                    new_program[j]['inputs'] = []
                    if j > 0:
                        stack = j-1
                elif is_double_arg(node['function']):
                    new_program[j]['inputs'] = [stack, j-1]
                else:
                    new_program[j]['inputs'] = [j-1]
            programs[i] = new_program

        elif funcs[-1] in equal_fns and (funcs[-2] == funcs[-3] and funcs[-2] in query_fns): # hacky but works
            _idx = funcs[1:].index('scene') + 1
            new_program = p[:_idx] + [p[-3]] + p[_idx:-3] + p[-2:]
            changed = True
        
        if changed:
            for j, node in enumerate(new_program):
                if node['function'] == 'scene':
                    new_program[j]['inputs'] = []
                    if j > 0:
                        stack = j-1
                elif is_double_arg(node['function']):
                    new_program[j]['inputs'] = [stack, j-1]
                else:
                    new_program[j]['inputs'] = [j-1]
            programs[i] = new_program

    return programs


# convert to universal representation from .structs
def clevr_formalize_programs(programs, concept_list=CLEVR_ATTRIBUTES.keys()):
    formal = []
    for p in programs:
        _nodes = []
        for i, node in enumerate(p):
            _fn_toks = node['function'].split('_')
            if len(_fn_toks) > 1 and _fn_toks[1] in concept_list:
                _fn, _concept = _fn_toks
            else:
                _fn = '_'.join(_fn_toks)
                _concept = None
            _value = None if not node['value_inputs'] else node['value_inputs'][0]
            _nodes.append(ProgramNode(step=i,
                                     function=_fn,
                                     inputs=node['inputs'],
                                     concept_input=_concept,
                                     value_input=_value,
            ))
        formal.append(_nodes)
    return formal