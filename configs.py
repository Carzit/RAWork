import os
import argparse
from numbers import Number
from typing import Callable, Literal, Union, Optional, List, Dict, Any


import utils

class Constraint:
    def __init__(self, attr_name:Optional[str] = None):
        self.attr_name:str = attr_name
    
    def check(self, value:Any):
        ...

    def attach(self, attr_name:Optional[str] = None):
        self.attr_name = attr_name

class ConstraintError(Exception):
    def __init__(self, message:str):
        self.message = message

class LambdaConstraint(Constraint):
    def __init__(self, func:Callable[[Any], bool], attr_name:Optional[str] = None):
        super().__init__(attr_name)
        self.func = func
    
    def check(self, value:Any):
        if not self.func(value):
            raise ConstraintError(f"{self.attr_name}: {value} does not satisfy the constraint")
        
    def __repr__(self):
        return f"LambdaConstraint({self.attr_name}, {self.func})"

class TypeConstraint(Constraint):
    def __init__(self, type_:type, attr_name:Optional[str] = None):
        super().__init__(attr_name)
        self._type = type_
    
    def check(self, value:Any):
        if not isinstance(value, self._type):
            raise ConstraintError(f"{self.attr_name}: {value} is not of type {self._type}") 
        
    def __repr__(self):
        return f"TypeConstraint({self.attr_name} should be type of {self._type})"

class EqualityConstraint(Constraint):
    def __init__(self, value:Any, attr_name:Optional[str] = None):
        super().__init__(attr_name)
        self.value = value
    
    def check(self, value:Any):
        if value != self.value:
            raise ConstraintError(f"{self.attr_name}: {value} is not equal to {self.value}")
        
    def __repr__(self):
        return f"EqualityConstraint({self.attr_name} = {self.value})"

class RangeConstraint(Constraint):
    def __init__(self, min_:Optional[float] = None, max_:Optional[float] = None, attr_name:Optional[str] = None):
        super().__init__(attr_name)
        self.min = min_
        self.max = max_
    
    def check(self, value:Number):
        if self.min is not None and value < self.min:
            raise ConstraintError(f"{self.attr_name}: {value} is less than minimum {self.min}")
        elif self.max is not None and value > self.max:
            raise ConstraintError(f"{self.attr_name}: {value} is greater than maximum {self.max}")
        
    def __repr__(self):
        return f"RangeConstraint({self.attr_name} ∈ [{self.min, self.max}])"
        
class LengthConstraint(Constraint):
    def __init__(self, min_:Optional[int] = None, max_:Optional[int] = None, attr_name:Optional[str] = None):
        super().__init__(attr_name)
        self.min = min_
        self.max = max_
    
    def check(self, value:List):
        if self.min is not None and len(value) < self.min:
            raise ConstraintError(f"{self.attr_name}: {value} has less than minimum length {self.min}")
        elif self.max is not None and len(value) > self.max:
            raise ConstraintError(f"{self.attr_name}: {value} has greater than maximum length {self.max}")
        
    def __repr__(self):
        return f"LengthConstraint(len{self.attr_name} ∈ [{self.min, self.max}])"
        
class ChoiceConstraint(Constraint):
    def __init__(self, choices:List[Any]=[], attr_name:Optional[str] = None):
        super().__init__(attr_name)
        self.choices = choices
    
    def check(self, value:Any):
        if value not in self.choices:
            raise ConstraintError(f"{self.attr_name}: {value} is not in choices {self.choices}")
        
    def __repr__(self):
        return f"ChoicesConstraint(len{self.attr_name} ∈ {self.choices}])"
        
class UnionConstraint(Constraint):
    def __init__(self, constraints:List[Constraint], attr_name:Optional[str] = None):
        super().__init__(attr_name)
        self.constraints = constraints
        for constraint in self.constraints:
            constraint.attach(attr_name)
    
    def check(self, value:Any):
        for constraint in self.constraints:
            try:
                constraint.check(value)
            except ConstraintError as e:
                raise e
            
class OrConstraint(Constraint):
    def __init__(self, constraints:List[Constraint], attr_name:Optional[str] = None):
        super().__init__(attr_name)
        self.constraints = constraints
        for constraint in self.constraints:
            constraint.attach(attr_name)
    
    def check(self, value:Any):
        for constraint in self.constraints:
            try:
                constraint.check(value)
                return
            except ConstraintError:
                pass
        raise ConstraintError(f"{self.attr_name}: {value} does not satisfy any of the constraints:{self.constraints}")

class NoConstraint(Constraint):
    def __init__(self, attr_name:Optional[str] = None):
        super().__init__(attr_name)
    
    def check(self, value:Any):
        pass
        
    def __repr__(self):
        return f"NoConstraint(len{self.attr_name})"
    
class NoneConstraint(Constraint):
    def __init__(self, attr_name:Optional[str] = None):
        super().__init__(attr_name)
    
    def check(self, value:Any):
        if value is not None:
            raise ConstraintError(f"{self.attr_name}: {value} is not None")
        
    def __repr__(self):
        return f"NoneConstraint(len{self.attr_name})"
    
class OptionalConstraint(Constraint):
    def __init__(self, constraint:Constraint, attr_name:Optional[str] = None):
        super().__init__(attr_name)
        self.constraint = constraint
        self.constraint.attach(attr_name)
    
    def check(self, value:Any):
        if value is not None:
            self.constraint.check(value)
        
    def __repr__(self):
        return f"OptionalConstraint({self.constraint})"


class Config:
    def __init__(self):
        self.valid_keys:List[str] = []
        self.key_constraints:Dict[str, Constraint] = {}
        self.config_dict:Dict[str, Any] = {}

    def __getitem__(self, k):
        if k in self.config_dict:
            return self.config_dict[k]
        else:
            raise KeyError(f"{k} not found in Config")

    def __setitem__(self, k, v):
        if k in self.config_dict:
            self.config_dict[k] = v
        else:
            raise KeyError(f"{k} not found in ConfigDict")
        
    def __getattr__(self, k):
        if k in ["valid_keys", "key_constraints", "config_dict"]:
            super().__getattr__(k)
        elif k in self.config_dict:
            return self.config_dict[k]
        else:
            raise AttributeError(f"Attribute {k} not found in Config")

    def __setattr__(self, k, v):
        if k in ["valid_keys", "key_constraints", "config_dict"]:
            super().__setattr__(k, v)
        elif k in self.config_dict:
            self.config_dict[k] = v
        else:
            raise AttributeError(f"Attribute {k} not found in Config")

    def __repr__(self):
        return f"Config({self.config_dict})"

    def from_file(self, config_file:str):
        configs:Dict[str, Any] = utils.read_configs(config_file=config_file)
        # check if all keys are valid
        for key in self.valid_keys:
            if key in configs:
                self.key_constraints[key].check(configs[key])
                self.config_dict[key] = configs[key]

    def from_args(self, args:Union[argparse.Namespace, argparse.ArgumentParser]):
        if isinstance(args, argparse.ArgumentParser):
            args = args.parse_args()
        for key in self.valid_keys:
            if hasattr(args, key):
                self.key_constraints[key].check( getattr(args, key))
                self.config_dict[key] = getattr(args, key)

    def from_kwargs(self, **kwargs):
        for key in self.valid_keys:
            if key in kwargs:
                self.key_constraints[key].check(kwargs[key])
                self.config_dict[key] = kwargs[key]

    def save(self, save_file:str):
        utils.save_configs(config_dict=self.config_dict, config_file=save_file)

class TokenizerConfig(Config):
    def __init__(self):
        super().__init__()
        self.valid_keys = ["vocab_file", "pretrained_tokenizer_file", "alias_file"]
        self.key_constraints = {"vocab_file": OptionalConstraint(TypeConstraint(str)),
                                "pretrained_tokenizer_file": OptionalConstraint(TypeConstraint(str)),
                                "alias_file": TypeConstraint(str)}
        for k, v in self.key_constraints.items():
            v.attach(k)
        self.config_dict = {"vocab_file": None, 
                            "pretrained_tokenizer_file": None, 
                            "alias_file": None}

class DataLoaderConfig(Config):
    def __init__(self):
        super().__init__()
        self.valid_keys = ["data_dir", "max_length", "batch_size", "shuffle", "num_workers", "pin_memory"]
        self.key_constraints = {"data_dir": TypeConstraint(str),
                                "max_length": TypeConstraint(int),
                                "batch_size": TypeConstraint(int), 
                                "shuffle": TypeConstraint(bool), 
                                "num_workers": TypeConstraint(int), 
                                "pin_memory": TypeConstraint(bool)}
        for k, v in self.key_constraints.items():
            v.attach(k)
        self.config_dict = {"data_dir": "data",
                            "max_length": 50,
                            "batch_size": 32, 
                            "shuffle": True, 
                            "num_workers": 0, 
                            "pin_memory": False}

class OptimizerConfig(Config):
    def __init__(self):
        super().__init__()
        self.valid_keys = ["optimizer_type", "optimizer_kwargs", "learning_rate", "lr_scheduler_type", "lr_scheduler_warmup_steps", "lr_scheduler_train_steps", "lr_scheduler_num_cycles", "lr_scheduler_power"]
        self.key_constraints = {"optimizer_type": ChoiceConstraint(["Adam", "Adam8bit", "PagedAdam8bit", "AdamW", "AdamW8bit", "PagedAdamW8bit", "Lion", "Lion8bit", "PagedLion8bit", "SGDNesterov","SGD8bit", "DAdaptation", "Adafactor"]), 
                                "optimizer_kwargs": NoConstraint(), 
                                "learning_rate": TypeConstraint(float), 
                                "lr_scheduler_type": ChoiceConstraint(["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "adafactor"]), 
                                "lr_scheduler_warmup_steps": TypeConstraint(int), 
                                "lr_scheduler_train_steps": OptionalConstraint(TypeConstraint(int)), 
                                "lr_scheduler_num_cycles": TypeConstraint(int), 
                                "lr_scheduler_power": TypeConstraint(int)}
        for k, v in self.key_constraints.items():
            v.attach(k)
        self.config_dict = {"optimizer_type": "PagedAdam8bit", 
                            "optimizer_kwargs": None, 
                            "learning_rate": 1e-3, 
                            "lr_scheduler_type": "cosine", 
                            "lr_scheduler_warmup_steps": 0, 
                            "lr_scheduler_train_steps": None, 
                            "lr_scheduler_num_cycles": 0, 
                            "lr_scheduler_power": 0}

class AssetBERTModelConfig(Config):
    def __init__(self):
        super().__init__()
        self.valid_keys = ["model_type","vocab_size","hidden_size","num_hidden_layers","num_attention_heads","intermediate_size","max_position_embeddings","type_vocab_size"]
        self.key_constraints = {"model_type": ChoiceConstraint(["BERT"]), 
                                "vocab_size": TypeConstraint(int), 
                                "hidden_size": TypeConstraint(int), 
                                "num_hidden_layers": TypeConstraint(int), 
                                "num_attention_heads": TypeConstraint(int), 
                                "intermediate_size": TypeConstraint(int), 
                                "max_position_embeddings": TypeConstraint(int), 
                                "type_vocab_size": TypeConstraint(int)}
        for k, v in self.key_constraints.items():
            v.attach(k)

        self.config_dict = {"model_type": "BERT",
                            "vocab_size": 30522,
                            "hidden_size": 768,
                            "num_hidden_layers": 12,
                            "num_attention_heads": 12,
                            "intermediate_size": 3072,
                            "max_position_embeddings": 512,
                            "type_vocab_size": 2}
        
class AssetBERTTrainConfig(Config):
    def __init__(self):
        super().__init__()
        self.valid_keys = ["max_epoches", "grad_clip_norm", "grad_clip_value", "detect_anomaly", "device", "dtype", "log_folder", "check_per_step", "report_per_epoch", "save_per_epoch", "save_folder", "save_name", "save_format"]
        self.key_constraints = {"max_epoches": TypeConstraint(int), 
                                "grad_clip_norm": OrConstraint([RangeConstraint(0, None), EqualityConstraint(-1)]),
                                "grad_clip_value": OrConstraint([RangeConstraint(0, None), EqualityConstraint(-1)]),
                                "gradient_accumulation_steps": UnionConstraint([TypeConstraint(int), RangeConstraint(1, None)]), 
                                "detect_anomaly": TypeConstraint(bool), 
                                "mixed_precision": ChoiceConstraint([None, 'no', 'fp16', 'bf16', 'fp8']),
                                "check_per_step": UnionConstraint([TypeConstraint(int), RangeConstraint(0, None)]), 
                                "report_per_epoch": UnionConstraint([TypeConstraint(int), RangeConstraint(0, None)]), 
                                "save_per_epoch": UnionConstraint([TypeConstraint(int), RangeConstraint(0, None)]), 
                                "save_folder": TypeConstraint(str), 
                                "save_name": TypeConstraint(str), 
                                "save_format": ChoiceConstraint([".pt", ".safetensors"])}
        for k, v in self.key_constraints.items():
            v.attach(k)

        self.config_dict = {"max_epoches": 40,
                            "grad_clip_norm": -1,
                            "grad_clip_value": -1,
                            "gradient_accumulation_steps": 1,
                            "detect_anomaly": True,
                            "mixed_precision": None,
                            "check_per_step": 300,
                            "report_per_epoch": 1,
                            "save_per_epoch": 1,
                            "save_folder": "model\\AttnFactorVAE\\test_softmax2",
                            "save_name": "AttnFactorVAE",
                            "save_format": ".pt"}
        

class ConfigDict:
    def __init__(self, configs:Dict[str, Config]):
        self._configs = configs
        self.valid_keys = list(configs.keys())

    def __getitem__(self, k):
        if k in self._configs:
            return self._configs[k]
        else:
            raise KeyError(f"Key {k} not found in ConfigDict")

    def __setitem__(self, k, v):
        if k in self._configs:
            self._configs[k] = v
        else:
            raise KeyError(f"Key {k} not found in ConfigDict")
        
    def __getattr__(self, k):
        if k in ["valid_keys", "_configs"]:
            super().__getattr__(k)
        elif k in self._configs:
            return self._configs[k]
        else:
            raise AttributeError(f"Attribute {k} not found in ConfigDict")

    def __setattr__(self, k, v):
        if k in ["valid_keys", "_configs"]:
            super().__setattr__(k, v)
        elif k in self._configs:
            self._configs[k] = v
        else:
            raise AttributeError(f"Attribute {k} not found in ConfigDict")

    def __repr__(self):
        return f"ConfigDict({self._configs})"
    
    def from_file(self, config_file:str):
        configs:Dict[str, Any] = utils.read_configs(config_file=config_file)
        # check if all keys are valid
        for key in self.valid_keys:
            if key in configs:
                self._configs[key].from_kwargs(**configs[key])

    def from_args(self, args:Union[argparse.Namespace, argparse.ArgumentParser]):
        if isinstance(args, argparse.ArgumentParser):
            args = args.parse_args()
        for key in self.valid_keys:
            self._configs[key].from_kwargs(**vars(args))

    def from_kwargs(self, **kwargs):
        for key in self.valid_keys:
            if key in kwargs:
                self._configs[key].from_kwargs(**kwargs[key])

    def save(self, save_file:str):
        config_dict = {k:v.config_dict for k,v in self._configs.items()}
        utils.save_configs(config_dict=config_dict, config_file=save_file)

    def save_separately(self, save_folder:str):
        for k, v in self._configs.items():
            v.save(f"{os.path.join(save_folder, k)}.json")
        
        