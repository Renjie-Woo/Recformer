from typing import List, Union
from transformers.models.longformer.modeling_longformer import LongformerConfig

# <s>, key, value, <pad>
PRETRAINED_LONGFORMER = 'allenai/longformer-base-4096'
DEFAULT_TOKEN_TYPE_SIZE = 4
DEFAULT_MAX_TOKEN_NUM = 1024
DEFAULT_MAX_ITEM_NUM = 51
DEFAULT_MAX_ATTR_NUM = 3,
DEFAULT_MAX_ATTR_LEN = 32

DEFAULT_TEMP = 0.05
DEFAULT_ITEM_NUM =22855
DEFAULT_FINETUNE_NEGATIVE_SAMPLE_SIZE = -1

class RecformerConfig(LongformerConfig):
    def __init__(self,
                 attention_window: Union[List[int], int] = 64,
                 sep_token_id: int = 2,
                 max_token_num: int = DEFAULT_MAX_TOKEN_NUM,
                 max_item_num: int = DEFAULT_MAX_ITEM_NUM,
                 max_attr_num: int = DEFAULT_MAX_ATTR_NUM,
                 max_attr_len: int = DEFAULT_MAX_ATTR_LEN,
                 temp: float = DEFAULT_TEMP,
                 item_num: int = DEFAULT_ITEM_NUM,
                 finetune_negative_sample_size: int = DEFAULT_FINETUNE_NEGATIVE_SAMPLE_SIZE,
                 **kwargs):
        super().__init__(attention_window, sep_token_id, **kwargs)

        self.token_type_size = DEFAULT_TOKEN_TYPE_SIZE
        self.max_token_num = max_token_num ## max length of input sequence
        self.max_item_num = max_item_num  ## max number of items in a sequence
        self.max_attr_num = max_attr_num  ## max number of attributes in an item
        self.max_attr_len = max_attr_len ## max length of an attribute
        self.temp = temp
        self.item_num = item_num
        self.finetune_negative_sample_size = finetune_negative_sample_size


DEFAULT_CONFIG = RecformerConfig.from_pretrained(PRETRAINED_LONGFORMER)
DEFAULT_CONFIG.max_attr_num = 3
DEFAULT_CONFIG.max_attr_length = 32
DEFAULT_CONFIG.max_item_num = 51
DEFAULT_CONFIG.max_token_num = 1024
DEFAULT_CONFIG.temp = DEFAULT_TEMP
DEFAULT_CONFIG.item_num = DEFAULT_ITEM_NUM
DEFAULT_CONFIG.finetune_negative_sample_size = DEFAULT_FINETUNE_NEGATIVE_SAMPLE_SIZE

if __name__ == '__main__':
    print(DEFAULT_CONFIG)