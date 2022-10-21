CLEVR_ATTRIBUTES = {
    'color' : ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow'],
    'material' : ['rubber', 'metal'],
    'shape' : ['cube', 'cylinder', 'sphere'],
    'size' : ['large', 'small']
}

CLEVR_RELATIONS = ['front', 'behind', 'left', 'right']

CLEVR_FUNCTIONS = [
    'count',
    'equal',
    'equal_integer',
    'exist',
    'filter',
    'greater_than',
    'intersect',
    'less_than',
    'query',
    'relate',
    'same',
    'scene',
    'union',
    'unique'
]

CLEVR_DOUBLE_ARGUMENT_FUNCTIONS = [
    'equal',
    'equal_integer',
    'greater_than',
    'less_than',
    'intersect',
    'union'
]

CLEVR_ANSWER_CANDIDATES = {
    'count': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'equal': ['yes', 'no'],
    'exist': ['yes', 'no'],
    'greater_than': ['yes', 'no'],
    'less_than': ['yes', 'no'],
    'query': CLEVR_ATTRIBUTES,
    'same': ['yes', 'no']
}


