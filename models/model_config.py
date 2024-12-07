#
DEFAULT_TITLE_SIZE = 30
DEFAULT_BODY_SIZE = 40
UNKNOWN_TITLE_VALUE = [0] * DEFAULT_TITLE_SIZE
UNKNOWN_BODY_VALUE = [0] * DEFAULT_BODY_SIZE

DEFAULT_DOCUMENT_SIZE = 768

HEAD_NUM = 20
HEAD_DIM = 20


class hparams_nrms:
    # INPUT DIMENTIONS:
    title_size: int = DEFAULT_TITLE_SIZE
    history_size: int = 50
    # MODEL ARCHITECTURE
    head_num: int = HEAD_NUM
    head_dim: int = HEAD_DIM
    attention_hidden_dim: int = HEAD_NUM * HEAD_DIM
    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    news_output_dim = HEAD_NUM * HEAD_DIM
    learning_rate: float = 0.0001


# class hparams_nrms_docvec:
#     # INPUT DIMENTIONS:
#     title_size: int = DEFAULT_DOCUMENT_SIZE
#     history_size: int = 50
#     # MODEL ARCHITECTURE
#     head_num: int = 20
#     head_dim: int = 20
#     attention_hidden_dim: int = 200
#     # MODEL OPTIMIZER:
#     optimizer: str = "adam"
#     loss: str = "cross_entropy_loss"
#     dropout: float = 0.2
#     learning_rate: float = 0.0001
#     newsencoder_units_per_layer: list[int] = [512, 512, 512]
