from programs.metadata.clevr import *
from utils import load_clevr_scenes

import random 


class ClevrExecutor:
    """Symbolic program executor for CLEVR"""

    def __init__(self, dataset_cfg: Optional[Dict[str, str]]=None):
        self.attributes = CLEVR_ATTRIBUTES
        self.relations = CLEVR_RELATIONS
        self._modules = CLEVR_FUNCTIONS
        self.answer_candidates = CLEVR_ANSWER_CANDIDATES
        self.double_argument_fns = CLEVR_DOUBLE_ARGUMENT_FUNCTIONS
        self.modules = {}
        self._register_modules()

        if dataset_cfg is not None:
            self.scenes_dataset = {split: load_clevr_scenes(ds) for split, ds in dataset_cfg.items()}
    
    def run_dataset(self, program, image_index, split, *args, **kwargs):
        assert self.modules and self.scenes_dataset, 'Must define modules and set scenes dataset first'
        assert split in self.scenes_dataset.keys(), f'invalid split {split}'
        scene = self.scenes_dataset[split][image_index]
        return self.run(program, scene, *args, **kwargs)

    def run(self, program, scene, guess=False, debug=False):
        assert self.modules, 'Must define modules first'
        assert type(scene) == list and type(scene[0]) == dict, 'scene is list of object dictionaries'
        
        ans, temp = None, None

        self.exe_trace = []
        for token in program:
            if token == 'scene':
                if temp is not None:
                    ans = 'error'
                    break
                temp = ans
                ans = list(scene)
            elif token in self.modules:
                module = self.modules[token]
                if token.startswith('same') or token.startswith('relate'):
                    ans = module(ans, scene)
                elif token in self.double_argument_fns:
                    ans = module(temp, ans)
                else:
                    ans = module(ans, temp)
                if ans == 'error':
                    break
            self.exe_trace.append(ans)
            if debug:
                print(token)
                print('ans:')
                self._print_debug_message(ans)
                print('temp: ')
                self._print_debug_message(temp)
                print()
        ans = str(ans)

        if ans == 'error' and guess:
            final_module = program[-1]
            if final_module == 'query':
                ans = random.choice(self.answer_candidates['query'][final_module])
            elif final_module in self.answer_candidates:
                ans = random.choice(self.answer_candidates[final_module])
        return ans

    def _print_debug_message(self, x):
        if type(x) == list:
            for o in x:
                print(self._object_info(o))
        elif type(x) == dict:
            print(self._object_info(x))
        else:
            print(x)

    def _object_info(self, obj):
        return '%s %s %s %s at %s' % (obj['size'], obj['color'], obj['material'], obj['shape'], str(obj['position']))
    
    def _register_modules(self):
        self.modules['count'] = self.count
        for attribute, attribute_values in self.attributes.items():
            self.modules[f'equal_{attribute}'] = self.equal(attribute)
            self.modules[f'query_{attribute}'] = self.query(attribute)
            self.modules[f'same_{attribute}'] = self.same(attribute)
            for value in attribute_values:
                self.modules[f'filter_{attribute}[{value}]'] = self.filter(attribute, value)
        for relation in self.relations:
            self.modules[f'relate[{relation}]'] = eval('self.relate_{relation}')
        self.modules['equal_integer'] = self.equal_integer
        self.modules['exist'] = self.exist
        self.modules['greater_than'] = self.greater_than
        self.modules['less_than'] = self.less_than
        self.modules['intersect'] = self.intersect
        self.modules['union'] = self.union
        self.modules['unique'] = self.unique
        
    def count(self, scene, _):
        if type(scene) == list:
            return len(scene)
        return 'error'
    
    def equal(self, attribute):
        assert attribute in self.attributes.keys(), f'unknown attribute {attribute}'
        vocab = self.attributes[attribute]
        def _equal(self, value1, value2):
            if type(value1) == str and value1 in vocab and type(value2) == str and value2 in vocab:
                if value1 == value2:
                    return 'yes'
                else:
                    return 'no'
            return 'error'
        return _equal 

    def equal_integer(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 == integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def exist(self, scene, _):
        if type(scene) == list:
            if len(scene) != 0:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def filter(self, attribute, value):
        assert attribute in self.attributes.keys(), f'unknown attribute {attribute}'
        assert value in self.attributes[attribute], f'unknown concept {value}'
        def _filter(self, scene, _):
            if type(scene) == list:
                output = []
                for o in scene:
                    if o[attribute] == value:
                        output.append(o)
                return output
            return 'error'
        return _filter
          
    def greater_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 > integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def less_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 < integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def intersect(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = []
            for o in scene1:
                if o in scene2:
                    output.append(o)
            return output
        return 'error'
    
    def query(self, attribute):
        assert attribute in self.attributes.keys(), f'unknown attribute {attribute}'
        def _query(self, obj, _):
            if type(obj) == dict and attribute in obj:
                return obj[attribute]
            return 'error'
        return _query

    def relate_behind(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][1] < obj['position'][1]:
                    output.append(o)
            return output
        return 'error'
    
    def relate_front(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][1] > obj['position'][1]:
                    output.append(o)
            return output
        return 'error'
    
    def relate_left(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][0] < obj['position'][0]:
                    output.append(o)
            return output
        return 'error'
    
    def relate_right(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][0] > obj['position'][0]:
                    output.append(o)
            return output
        return 'error'
    
    def same(self, attribute):
        assert attribute in self.attributes.keys(), f'unknown attribute {attribute}'
        def _same(self, obj, scene):
            if type(obj) == dict and attribute in obj and type(scene) == list:
                output = []
                for o in scene:
                    if o[attribute] == obj[attribute] and o['id'] != obj['id']:
                        output.append(o)
                return output
            return 'error'
        return _same

    def union(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = list(scene2)
            for o in scene1:
                if o not in scene2:
                    output.append(o)
            return output
        return 'error'
    
    def unique(self, scene, _):
        if type(scene) == list and len(scene) > 0:
            return scene[0]
        return 'error'