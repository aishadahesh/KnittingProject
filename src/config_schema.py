from typing import List, Dict
from pydantic import BaseModel

class UIConfig(BaseModel):
    reference_image: str

class GeometryParameter(BaseModel):
    name: str
    initial: float
    range: List[float]
    delta: float

class KnitParameters(BaseModel):
    yarn_colors: List[List[float]]
    bitmap_rows: int
    bitmap_loops: int
    loop_res: int
    segments: int
    parameters: List[GeometryParameter]

class RenderingConfig(BaseModel):
    mitsuba_variant: str
    mitsuba_variant_fallback: str
    output_dir: str
    camera_dist_mult: float
    camera_fov: float
    spp_optimization: int

class OptimizationConfig(BaseModel):
    learning_rate: float
    loss_weights: Dict[str, float]
    loss_center_crop: List[float]
    max_epochs: int
    patience: int

class AppConfig(BaseModel):
    ui: UIConfig
    knit_parameters: KnitParameters
    rendering: RenderingConfig
    optimization: OptimizationConfig
