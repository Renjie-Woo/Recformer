import torch
from transformers import LongformerTokenizer
from recformer_v2.config import DEFAULT_CONFIG

TOKEN_TYPE_CLS = 0
TOKEN_TYPE_KEY = 1
TOKEN_TYPE_VALUE = 2
TOKEN_TYPE_PAD = 3


class RecformerTokenizer(LongformerTokenizer):
    def __init__(
            self,
            vocab_file,
            merges_file,
            config=None,
            errors="replace",
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            add_prefix_space=False,
            **kwargs,
    ):
        if config is None:
            self.config = DEFAULT_CONFIG
        else:
            self.config = config

        super().__init__(vocab_file, merges_file, errors=errors, bos_token=bos_token, \
                         eos_token=eos_token, sep_token=sep_token, cls_token=cls_token, \
                         unk_token=unk_token, pad_token=pad_token, mask_token=mask_token, \
                         add_prefix_space=add_prefix_space, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        cls.config = DEFAULT_CONFIG
        return super().from_pretrained(pretrained_model_name_or_path)

    def __call__(self, items, pad_to_max=False, return_tensor=False):
        if items is None:
            raise ValueError('You need to specify the items.')
        if not isinstance(items, list):
            raise TypeError('Items must be type of list[obj].')
        _len_ = len(items)
        if _len_ < 1:
            return ValueError('Items must contain one item information in it.')
        if isinstance(items[0], list):
            result = self.batch_encode(items, pad_to_max)
        else:
            result = self.encode(items)

        if return_tensor:
            for k,v in result.items():
                result[k] = torch.LongTensor(v)
        return result

    def set_config(self, config=DEFAULT_CONFIG):
        self.config = config

    def _text_tokenize(self, text):
        """convert string to token ids"""
        return self.convert_tokens_to_ids(self.tokenize(text))

    def _attribute_encode(self, attr_name, attr_value):
        """
  encode attribute
  """
        # title: Apple Music
        # title -> [title] -> [425326]
        attr_name_token_ids = self._text_tokenize(attr_name)
        # Apple Music -> [Apple, Music] -> [122, 73873]
        attr_value_token_ids = self._text_tokenize(attr_value)

        # [1]
        key_token_tags = [TOKEN_TYPE_KEY] * len(attr_name_token_ids)
        # [2, 2]
        value_token_tags = [TOKEN_TYPE_VALUE] * len(attr_value_token_ids)

        attr_token_ids, token_tags = list(), list()
        attr_token_ids.extend(attr_name_token_ids)
        attr_token_ids.extend(attr_value_token_ids)

        token_tags.extend(key_token_tags)
        token_tags.extend(value_token_tags)

        # input_ids: [425326, 122, 73873]
        # token_ids: [1, 2, 2]
        return attr_token_ids[:self.config.max_attr_len], token_tags[:self.config.max_attr_len]

    def encode_item(self, item):
        """
  item: {"key1":"value1", "key2":"vale2", ...}

  """
        input_ids = []
        token_type_ids = []

        # token_type_ids: [1, 2, 2, 1, 2, 1, 2, 2, 2]
        for attr_name, attr_value in item.items():
            attr_token_ids, token_tags = self._attribute_encode(attr_name, attr_value)
            input_ids.extend(attr_token_ids)
            token_type_ids.extend(token_tags)

        return input_ids[:self.config.max_attr_len], token_type_ids[:self.config.max_attr_len]

    def encode(self, items):
        items = items[::-1][:self.config.max_item_num]

        input_ids = [self.bos_token_id]
        item_position_ids = [0]
        token_type_ids = [TOKEN_TYPE_CLS]

        for index, item in enumerate(items):
            # token_type_ids: [1, 2, 2, 1, 2, 1, 2, 2, 2]
            _input_ids, _token_type_ids = self.encode_item(item)
            input_ids.extend(_input_ids)
            token_type_ids.extend(_token_type_ids)
            item_position_ids.extend([index + 1] * len(_input_ids))

        input_ids = input_ids[:self.config.max_token_num]
        item_position_ids = item_position_ids[:self.config.max_token_num]
        token_type_ids = token_type_ids[:self.config.max_token_num]

        attention_mask = [1] * len(input_ids)
        global_attention_mask = [0] * len(input_ids)
        global_attention_mask[0] = 1

        return {
            "input_ids": input_ids,
            "item_position_ids": item_position_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask
        }

    def padding(self, enc_items, pad_to_max):
        if pad_to_max:
            max_len = self.config.max_token_num
        else:
            max_len = max([len(item['input_ids']) for item in enc_items])

        batch_input_ids = []
        batch_item_position_ids = []
        batch_token_type_ids = []
        batch_attention_mask = []
        batch_global_attention_mask = []

        for item in enc_items:
            input_ids = item['input_ids']
            item_position_ids = item['item_position_ids']
            token_type_ids = item['token_type_ids']
            attention_mask = item['attention_mask']
            global_attention_mask = item['global_attention_mask']

            pad_len = max_len - len(input_ids)

            input_ids.extend([self.pad_token_id] * pad_len)
            item_position_ids.extend([-1] * pad_len)
            token_type_ids.extend([TOKEN_TYPE_PAD] * pad_len)
            attention_mask.extend([0] * pad_len)
            global_attention_mask.extend([0] * pad_len)

            batch_input_ids.append(input_ids)
            batch_item_position_ids.append(item_position_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_global_attention_mask.append(global_attention_mask)

        return {
            "input_ids": batch_input_ids,
            "item_position_ids": batch_item_position_ids,
            "token_type_ids": batch_token_type_ids,
            "attention_mask": batch_attention_mask,
            "global_attention_mask": batch_global_attention_mask
        }

    def batch_encode(self, batch_items, pad_to_max=False):
        batch_items = [self.encode(item) for item in batch_items]

        return self.padding(batch_items, pad_to_max)


if __name__ == '__main__':
    tokenizer = RecformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    # tokenizer.set_config()

    items1 = [{'pt': 'PUZZLES',
               'material': 'Cardboard++Cartón',
               'item_dimensions': '27 x 20 x 0.1 inches',
               'number_of_pieces': '1000',
               'brand': 'Galison++',
               'number_of_items': '1',
               'model_number': '9780735366763',
               'size': '1000++',
               'theme': 'Christmas++',
               'color': 'Dresden'},
              {'pt': 'DECORATIVE_SIGNAGE',
               'item_shape': 'Square++Cuadrado',
               'brand': 'Generic++',
               'color': 'Square-5++Cuadrado-5',
               'mounting_type': 'Wall Mount++',
               'material': 'Wood++Madera'}]
    inputs = tokenizer(items1)
    print(inputs)

    inputs = tokenizer(items1, return_tensor=True)
    print(inputs)


    items2 = [{'pt': 'WALL_ART',
               'number_of_items': '1',
               'mounting_type': 'Wall Mount++',
               'item_shape': 'Rectangular++',
               'brand': "Teacher's Discovery++",
               'color': '_++'},
              {'pt': 'CALENDAR',
               'theme': 'Funny, Love, Wedding++',
               'format': 'wall_calendar',
               'model_year': '2022',
               'brand': 'CALVENDO++',
               'size': 'Square++cuadrado',
               'material': 'Paper, Wool++'},
              {'pt': 'BLANK_BOOK',
               'number_of_items': '1',
               'color': 'Hanging Flowers++Flores colgantes',
               'brand': 'Graphique++',
               'ruling_type': 'Ruled++',
               'binding': 'office_product',
               'paper_size': '6.25 x 8.25 inches++',
               'style': 'Hanging Flowers'}]
    inputs = tokenizer([items1, items2])
    print(inputs)

    inputs = tokenizer([items1, items2], pad_to_max=True)
    print(inputs)

    inputs = tokenizer([items1, items2], return_tensor=True)
    print(inputs)