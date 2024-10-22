from dataclasses import dataclass, field, fields
from typing import Optional

@dataclass
class Config:
    experiment_name: Optional[str] = "seq2seq4"
    root_dir: Optional[str] = "./"
    device: Optional[str] = "cuda:1"
        
    #training parameters
    epochs: Optional[int] = 10
    seed: Optional[int] = 42
    use_half_precision: Optional[bool] = True

    # scheduler parameters
    scheduler_type: Optional[str] = "cosine_annealing_warm_restart" # multi_step or none
    T_0: Optional[int] = 10
    T_mult: Optional[int] = 1

    # optimizer parameters
    optimizer_type: Optional[str] = "adam" # sgd or adam
    optimizer_lr: Optional[float] = 0.0001   
    optimizer_momentum: Optional[float] = 0.9
    optimizer_weight_decay: Optional[float] = 0.0001
    optimizer_no_decay: Optional[list] = field(default_factory=list)
    clip_grad_norm: Optional[float] = -1
        
    # Model Parameters
    model_name: Optional[str] = "seq2seq_transformer"
    hybrid: Optional[bool] = True
    embedding_size: Optional[int] = 64
    hidden_dim: Optional[int] = 64
    pff_dim: Optional[int] = 512
    nhead: Optional[int] = 8
    num_encoder_layers: Optional[int] = 1
    num_decoder_layers: Optional[int] = 1
    dropout: Optional[int] = 0.2
    pretrain: Optional[bool] = False
    input_emb_size: Optional[int] = 64
    max_input_points: Optional[int] = 33
    src_vocab_size: Optional[int] = 3
    tgt_vocab_size: Optional[int] = 59

    # Criterion
    criterion: Optional[str] = "cross_entropy"
        
    def print_config(self):
        print("="*50+"\nConfig\n"+"="*50)
        for field in fields(self):
            print(field.name.ljust(30), getattr(self, field.name))
        print("="*50)

    def save(self, root_dir):
        path = root_dir + "/config.txt"
        with open(path, "w") as f:
            f.write("="*50+"\nConfig\n"+"="*50 + "\n")
            for field in fields(self):
                f.write(field.name.ljust(30) + ": " + str(getattr(self, field.name)) + "\n")
            f.write("="*50)   
