from typing import List, Union
from transformers.models.longformer.modeling_longformer import LongformerConfig

# <s>, key, value, <pad>
PRETRAINED_LONGFORMER = 'allenai/longformer-base-4096'
DEFAULT_TOKEN_TYPE_SIZE = 4
DEFAULT_MAX_TOKEN_NUM = 2048
DEFAULT_MAX_ITEM_NUM = 64
DEFAULT_MAX_ATTR_NUM = 3,
DEFAULT_MAX_ATTR_LEN = 32


class RecformerConfig(LongformerConfig):
    def __init__(self,
                 max_token_num: int = DEFAULT_MAX_TOKEN_NUM,
                 max_item_num: int = DEFAULT_MAX_ITEM_NUM,
                 max_attr_num: int = DEFAULT_MAX_ATTR_NUM,
                 max_attr_len: int = DEFAULT_MAX_ATTR_LEN,
                
                 ):
        super().__init__()

        self.token_type_size = DEFAULT_TOKEN_TYPE_SIZE
        self.max_token_num = max_token_num ## max length of input sequence
        self.max_item_num = max_item_num  ## max number of items in a sequence
        self.max_attr_num = max_attr_num  ## max number of attributes in an item
        self.max_attr_len = max_attr_len ## max length of an attribute

    

DEFAULT_CONFIG = RecformerConfig()

if __name__ == '__main__':
    print(DEFAULT_CONFIG)