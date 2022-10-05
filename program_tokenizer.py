from typings import *

import torch
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
    '<{>': 4,
    '<}>': 5,
    '<[>': 6,
    '<]>': 7,
}

CLEVR_DOUBLE_ARG_PRIMITIVES = [
    'equal_color',
    'equal_integer',
    'equal_material',
    'equal_shape',
    'equal_size',
    'greater_than',
    'intersect',
    'less_than',
    'union'
]


Tokens = List[str]
Vocabulary = Dict[str, int]


class ProgramTokenizer:

    special_tokens = SPECIAL_TOKENS 
    pad_token, pad_token_id = '<PAD>', 0    # Padding
    sos_token, sos_token_id = '<START>', 1  # Start of Sequence
    eos_token, eos_token_id = '<END>', 2    # End of Sequence
    unk_token, unk_token_id = '<UNK>', 3    # unknown token 
    sca_token, sca_token_id = '<{>', 4     # concept-agnostic: start concept argument
    eca_tokeb, eca_token_id = '<}>', 5    # concept-agnostic: end concept argument
    sva_token, sva_token_id = '<[>', 6     # vocab-agnostic: start value argument
    eva_tokeb, eva_token_id = '>]>', 7    # vocab-agnostic: end value argument
    start_exec_primitive = 'scene'
    double_argument_nodes = CLEVR_DOUBLE_ARG_PRIMITIVES
    
    def make_vocab(self, vocab: Dict[str, int]):
        self.vocab = vocab 
        self.vocab_inv = {v:k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def make_from_dataset(self, programs: List[Program], reverse: bool = True) -> List[Tokens]:
        tokens = self.preprocess_programs(programs, reverse=reverse)
        
        all_tokens = set()
        for t in tokens:
            all_tokens = all_tokens.union(set(t))

        prog_vocab = {
            **self.special_tokens,
            **{v: k + len(self.special_tokens) for k, v in enumerate(sorted(all_tokens))},
        }

        self.make_vocab(prog_vocab)

        return tokens

    def _parse_fn_from_str(self, node: str) -> str:
        return node.split('(')[0].split('[')[0].split('{')[0]

    def _parse_concept_from_str(self, node: str) -> Optional[str]:
        return None if len(node.split('{')) < 2 else node.split('{')[1].split('}')[0]

    def _parse_value_from_str(self, node: str) -> Optional[str]:
        return None if len(node.split('[')) < 2 else node.split('[')[1].split(']')[0]

    # def _parse_inputs_from_str(self, node: str) -> List[int]:
    #     try:
    #         return [] if '()' in node else list(map(int, node.split('(')[1].split(')')[0].split(',')))
    #     except IndexError:
    #         print(node)

    def _parse_inputs_from_str(self, node: str, step: int) -> List[int]:
        # if '()' in node:
        #     return []
        # if node in self.double_argument_nodes:
        #     return [step - 1, stack]
        if node == self.start_exec_primitive:
            return []
        # -1 represents the stack, from previous branch
        return [step - 1] if node not in self.double_argument_nodes else [step-1, -1]


    def _convert_token_to_node(self, node: str, step: int) -> ProgramNode:
        return ProgramNode(step=step,
            function=self._parse_fn_from_str(node),
            inputs=self._parse_inputs_from_str(node, step),
            #inputs=[] if not step else [step-1],
            concept_input=self._parse_concept_from_str(node),
            value_input=self._parse_value_from_str(node)
        )

    def tokenize(self, program: Program) -> Tokens:
        return [
                 node.function
                 + ("" if not node.concept_input else "{" + node.concept_input + "}")
                 + ("" if not node.value_input else "[" + node.value_input + "]")
                 for node in program
             ]

    def convert_programs_to_tokens(self, programs: List[Program]) -> List[Tokens]:
        return list(map(self.tokenize, programs))

    def convert_tokens_to_ids(self, tokens: List[Tokens]) -> List[int]:
        return [ 
            [
                 self.vocab[t] for t in p
             ]
             for p in tokens
        ]

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[Tokens]:
        return [ 
            [
                 self.vocab_inv[t] for t in p
             ]
             for p in token_ids
        ]

    def convert_tokens_to_programs(self, tokens: List[Tokens]) -> List[Program]:
        return [ 
            [
                 self._convert_token_to_node(node, i) 
                 for i, node in enumerate(p)
             ]
             for p in tokens
        ]

    def convert_ids_to_programs(self, token_ids: List[int]) -> List[Program]:
        return [ 
            [
                 self._convert_token_to_node(self.vocab_inv[t], i)
                 for i, t in enumerate(p)
             ]
             for p in token_ids
        ]

    def convert_programs_to_ids(self, programs: List[Program]) -> List[int]:
        return [ 
            [
                 self.vocab[node.function
                 + ("" if not node.concept_input else "{" + node.concept_input + "}")
                 + ("" if not node.value_input else "[" + node.value_input + "]")]
                 for node in p
             ]
             for p in programs
        ]

    def _clevr_convert_to_chain(self, programs: List[Tokens]) -> List[Tokens]:
        # change the order of the functions so next step takes always input of previous step
        # except two-argument primitives, that will use a stack
        # in CLEVR this can be fixed manually.
        # @TODO: fix it for any given dataset using the inputs field to determine swap_id
        programs = programs.copy()
        query_fns = ['query{color}', 'query{shape}', 'query{size}' 'query{material}']
        equal_fns = ['equal{color}', 'equal{shape}', 'equal{size}' 'equal{material}']
        for i, p in enumerate(programs):
            if p[-1] in equal_fns and (p[-2] == p[-3] and p[-2] in query_fns):
                swap_id = p[1:].index('scene') + 1 
                programs[i] = p[:swap_id] + [p[-3]] + p[swap_id:-3] + p[-2:]
        return programs

    def _reverse(self, programs: List[Tokens]) -> List[Tokens]:
        return [p[::-1] for p in programs]

    def preprocess_programs(self, programs: List[Program], reverse: bool = True) -> List[Tokens]:
        tokens = self._clevr_convert_to_chain(self.convert_programs_to_tokens(programs))
        return self._reverse(tokens) if reverse else tokens

    def preprocess_tokens(self, tokens: List[Tokens], reverse: bool = True) -> List[Tokens]:
        tokens = self._clevr_convert_to_chain(tokens)
        return self._reverse(tokens) if reverse else tokens

    def encode(self, tokens: Tokens, max_len: Optional[int] = None) -> Tensor:
        tokens = self.preprocess_tokens([tokens])[0]
        token_ids = torch.tensor(
                [self.vocab["<START>"]]
                + [
                    self.vocab[t] if t in self.vocab.keys() else self.vocab["<UNK>"]
                    for t in tokens
                ]
                + [self.vocab["<END>"]],
                dtype=torch.long
            )
        if max_len is not None:
            token_ids = F.pad(token_ids, (0, max_len-token_ids.shape[0]))
        return token_ids

    def batch_encode(self, 
                     tokens: List[Tokens], 
                     max_len: Optional[int] = None
    ) -> Tensor:
        tokens = self.preprocess_tokens(tokens)
        token_ids = pad_sequence([
                torch.tensor(
                    [self.vocab["<START>"]]
                    + [
                        self.vocab[t] if t in self.vocab.keys() else self.vocab["<UNK>"]
                        for t in ts
                    ]
                    + [self.vocab["<END>"]],
                    dtype=torch.long,
                )
                for ts in tokens
            ], padding_value=self.pad_token_id, batch_first=True)
        if max_len is not None:
            token_ids = pad_sequence(token_ids + [torch.empty(max_len, dtype=torch.long)],
                padding_value=self.pad_token_id, batch_first=True)[:-1]
            assert token_ids.shape[0] == len(programs)
        return token_ids

    def encode_program(self, program: Program, max_len: Optional[int] = None) -> Tensor:
        tokens = self.preprocess_programs([program])[0]
        token_ids = torch.tensor(
                [self.vocab["<START>"]]
                + [
                    self.vocab[t] if t in self.vocab.keys() else self.vocab["<UNK>"]
                    for t in tokens
                ]
                + [self.vocab["<END>"]],
                dtype=torch.long
            )
        if max_len is not None:
            token_ids = F.pad(token_ids, (0, max_len-token_ids.shape[0]))
        return token_ids

    def batch_encode_program(self, 
                     programs: List[Program], 
                     max_len: Optional[int] = None
    ) -> Tensor:
        tokens = self.preprocess_programs(programs)
        token_ids = pad_sequence([
                torch.tensor(
                    [self.vocab["<START>"]]
                    + [
                        self.vocab[t] if t in self.vocab.keys() else self.vocab["<UNK>"]
                        for t in ts
                    ]
                    + [self.vocab["<END>"]],
                    dtype=torch.long,
                )
                for ts in tokens
            ], padding_value=self.pad_token_id, batch_first=True)
        if max_len is not None:
            token_ids = pad_sequence(token_ids + [torch.empty(max_len, dtype=torch.long)],
                padding_value=self.pad_token_id, batch_first=True)[:-1]
            assert token_ids.shape[0] == len(programs)
        return token_ids

    def decode(self, token_ids: Tensor) -> Program:
        # single sequence
        assert len(token_ids.shape) == 1
        return self.batch_decode(token_ids.unsqueeze(0))[0]

    def batch_decode(self, token_ids: Tensor) -> List[Tokens]:
        # batch of sequences
        assert len(token_ids.shape) == 2
        # remove pad, start and end tokens and re-reverse
        eos_mask = (torch.where(token_ids == self.eos_token_id)[1] + 1).tolist()
        token_ids = [toks[:idx][1:-1][::-1] for toks, idx in zip(token_ids.tolist(), eos_mask)]
        return self.convert_ids_to_tokens(token_ids)

    def decode_program(self, token_ids: Tensor) -> Program:
        # single sequence
        assert len(token_ids.shape) == 1
        return self.batch_decode_program(token_ids.unsqueeze(0))[0]

    def batch_decode_program(self, token_ids: Tensor) -> List[Program]:
        # batch of sequences
        assert len(token_ids.shape) == 2
        # remove pad, start and end tokens and re-reverse
        eos_mask = (torch.where(token_ids == self.eos_token_id)[1] + 1).tolist()
        token_ids = [toks[:idx][1:-1][::-1] for toks, idx in zip(token_ids.tolist(), eos_mask)]
        return self.convert_ids_to_programs(token_ids)

# def create_filter_dataset(programs: List[Program]):
#     dataset = {}
#     for pid, program in enumerate(programs):
#         dataset[pid] = []
#         ins, outs = [], []
#         idx = 0
#         previous = None
#         continuing = False
#         while idx < len(program):   
#             node_type = program[idx]['type']
#             if node_type == "scene":
#                 if previous is not None:
#                     dataset[pid].append(previous)
#                 previous = None
                
#             elif node_type.startswith("filter"):
#                 inp = program[idx]['value_inputs']
#                 out = program[idx]['_output']

#                 if previous is None:
#                     continuing = True
#                     previous = (inp, out)

#                 else:
#                     if continuing:
#                         previous = (previous[0] + inp, out)
#                     else:
#                         dataset[pid].append(previous)
#                         previous = None
#             else:
#                 continuing = False        
#             idx += 1
