from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    num_layers: int = Field(default = None, ge = 11, le = 101)
    num_classes: int = Field(gt = 0)
    small_dataset: bool = Field(default=False)
