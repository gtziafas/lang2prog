from structs import *

import torch
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = [
    '<PAD>',
    '<START>',
    '<END>',
    '<UNK>'
]

VALUE_ARGUMENT_TOKENS = [
    '<]>',
    '<[>'
]

CONCEPT_ARGUMENT_TOKENS = [
    '<}>',
    '<{>'
]

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

    pad_token, pad_token_id = '<PAD>', 0    # Padding
    sos_token, sos_token_id = '<START>', 1  # Start of Sequence
    eos_token, eos_token_id = '<END>', 2    # End of Sequence
    unk_token, unk_token_id = '<UNK>', 3    # unknown token 
    start_exec_primitive = 'scene'
    double_argument_nodes = CLEVR_DOUBLE_ARG_PRIMITIVES#

    def __init__(self, 
                 vocab: Optional[Vocabulary] = None, 
                 version: int = 0,
                 reverse: bool = True
    ):
        if vocab is not None:
            self.make_vocab(vocab)
        
        assert version in [0, 1, 2]
        if version == 0:
            self.tokenize = self._tokenize
            self.special_tokens = SPECIAL_TOKENS

        elif version == 1:
            self.tokenize = self._tokenize_v1
            self.special_tokens = SPECIAL_TOKENS + VALUE_ARGUMENT_TOKENS
            self.eva_token, self.sva_token = VALUE_ARGUMENT_TOKENS
            self.sva_token_id = self.special_tokens.index(self.sva_token), 
            self.eva_token_id = self.special_tokens.index(self.eva_token)

        elif version == 2:
            self.tokenize = self._tokenize_v2
            self.special_tokens = SPECIAL_TOKENS + VALUE_ARGUMENT_TOKENS + CONCEPT_ARGUMENT_TOKENS
            self.eva_token, self.sva_token = VALUE_ARGUMENT_TOKENS
            self.eca_token, self.sca_token = CONCEPT_ARGUMENT_TOKENS
            self.sva_token_id = self.special_tokens.index(self.sva_token), 
            self.eva_token_id = self.special_tokens.index(self.eva_token)
            self.sca_token_id = self.special_tokens.index(self.sca_token), 
            self.eca_token_id = self.special_tokens.index(self.eca_token)

        self.version = version
        self.reverse = reverse
    
    def make_vocab(self, vocab: Vocabulary):
        self.vocab = vocab 
        self.vocab_inv = {v:k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def make_from_dataset(self, programs: List[Program]) -> List[Tokens]:
        tokens = self.preprocess_programs(programs)

        if self.version:
            start_value, end_value = self.sva_token, self.eva_token
            if self.reverse:
                start_value, end_value = end_value, start_value
        
        all_tokens = set()
        for ts in tokens:
            for i, t in enumerate(ts):
                if self.version:
                    if (i > 0 and ts[i - 1] == start_value) and (i+1 < len(ts) and ts[i+1] == end_value): 
                        continue
                all_tokens.add(t)
        all_tokens = all_tokens.difference(set(self.special_tokens))

        prog_vocab = {
            **{v: k for k, v in enumerate(self.special_tokens)},
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

    def _parse_inputs_from_str(self, node: str, step: int) -> List[int]:
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

    def _convert_node_to_token(self, node: ProgramNode) -> str:
        _concept = ("" if not node.concept_input else "{" + node.concept_input + "}")
        _value = ("" if not node.value_input else "[" + node.value_input + "]")
        return  node.function + _concept + _value

    def _convert_to_v0(self, tokens: List[Tokens]) -> List[Tokens]:
        to_reverse = False
        if tokens[0][-2] == self.start_exec_primitive:
            # its reverse, change it
            to_reverse = True
            tokens = self._reverse(tokens)
        def _convert(ts: Tokens) -> Tokens:
            output_sequence = []
            i=0
            while i < len(ts):
                token = ts[i]
                if self.version == 2:
                    if token == self.sca_token:
                        output_sequence[-1] += '{' + ts[i+1] + '}'
                        i += 3
                        continue
                if token == self.sva_token:
                    output_sequence[-1] += '[' + ts[i+1] + ']'
                    i += 3
                    continue
                output_sequence.append(token)
                i += 1
            return output_sequence if not to_reverse else self._reverse([output_sequence])[0]
        return list(map(_convert, tokens))

    def _tokenize(self, program: Program) -> Tokens:
        return list(map(self._convert_node_to_token, program))

    def _tokenize_v1(self, program: Program) -> Tokens:
        output_sequence = []
        for i, node in enumerate(program):
            _concept = "" if not node.concept_input else "{" + node.concept_input + "}"
            output_sequence.append(node.function + _concept)
            if node.value_input is not None:
                output_sequence.extend([self.sva_token, node.value_input, self.eva_token])
        return output_sequence

    def _tokenize_v2(self, program: Program) -> Tokens:
        output_sequence = []
        for i, node in enumerate(program):
            output_sequence.append(node.function) 
            if node.concept_input is not None:
                output_sequence.extend([self.sca_token, node.concept_input, self.eca_token])
            if node.value_input is not None:
                output_sequence.extend([self.sva_token, node.value_input, self.eva_token])
        return output_sequence

    def convert_programs_to_tokens(self, programs: List[Program]) -> List[Tokens]:
        return list(map(self.tokenize, programs))

    def convert_tokens_to_programs(self, tokens: List[Tokens]) -> List[Program]:
        tokens = self._convert_to_v0(tokens) if self.version else tokens
        return [ 
            [
                 self._convert_token_to_node(token, i) 
                 for i, token in enumerate(p)
             ]
             for p in tokens
        ]

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

    def convert_ids_to_programs(self, token_ids: List[int]) -> List[Program]:
        return self.convert_tokens_to_programs(self.convert_ids_to_tokens(token_ids))

    def convert_programs_to_ids(self, programs: List[Program]) -> List[int]:
        return self.convert_tokens_to_ids(self.convert_programs_to_tokens(programs))

    def _reverse(self, tokens: List[Tokens]) -> List[Tokens]:
        return [ts[::-1] for ts in tokens]

    def preprocess_programs(self, programs: List[Program]) -> List[Tokens]:
        tokens = self.convert_programs_to_tokens(programs)
        return self._reverse(tokens) if self.reverse else tokens

    def preprocess_tokens(self, tokens: List[Tokens]) -> List[Tokens]:
        return self._reverse(tokens) if self.reverse else tokens

    def encode(self, tokens: Tokens, max_len: Optional[int] = None) -> Tensor:
        tokens = self.preprocess_tokens([tokens])[0]
        token_ids = torch.tensor(
                [self.sos_token_id]
                + [
                    self.vocab[t] if t in self.vocab.keys() else self.unk_token_id
                    for t in tokens
                ]
                + [self.eos_token_id],
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
                    [self.sos_token_id]
                    + [
                        self.vocab[t] if t in self.vocab.keys() else self.unk_token_id
                        for t in ts
                    ]
                    + [self.eos_token_id],
                    dtype=torch.long,
                )
                for ts in tokens
            ], padding_value=self.pad_token_id, batch_first=True)
        if max_len is not None:
            token_ids = pad_sequence(list(token_ids) + [torch.empty(max_len, dtype=torch.long)],
                padding_value=self.pad_token_id, batch_first=True)[:-1]
            assert token_ids.shape[0] == len(programs)
        return token_ids

    def encode_program(self, program: Program, max_len: Optional[int] = None) -> Tensor:
        tokens = self.preprocess_programs([program])[0]
        token_ids = torch.tensor(
                [self.sos_token_id]
                + [
                    self.vocab[t] if t in self.vocab.keys() else self.unk_token_id
                    for t in tokens
                ]
                + [self.eos_token_id],
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
                    [self.sos_token_id]
                    + [
                        self.vocab[t] if t in self.vocab.keys() else self.unk_token_id
                        for t in ts
                    ]
                    + [self.eos_token_id],
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
        token_ids = [toks[:idx][1:-1] for toks, idx in zip(token_ids.tolist(), eos_mask)]
        token_ids = [toks[::-1] for toks in token_ids] if self.reverse else token_ids
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
        token_ids = [toks[:idx][1:-1] for toks, idx in zip(token_ids.tolist(), eos_mask)]
        token_ids = [toks[::-1] for toks in token_ids] if self.reverse else token_ids
        return self.convert_ids_to_programs(token_ids)
