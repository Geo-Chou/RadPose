from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import (
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
)
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry

RUNNERS = Registry("runner", parent=MMENGINE_RUNNERS)

RUNNER_CONSTRUCTORS = Registry(
    "runner constructor", parent=MMENGINE_RUNNER_CONSTRUCTORS
)

LOOPS = Registry("loop", parent=MMENGINE_LOOPS)

HOOKS = Registry("hook", parent=MMENGINE_HOOKS, locations=["mmwave3d.hooks"])


DATASETS = Registry(
    "dataset", parent=MMENGINE_DATASETS, locations=["mmwave3d.datasets"]
)
DATA_SAMPLERS = Registry("data sampler", parent=MMENGINE_DATA_SAMPLERS)
TRANSFORMS = Registry(
    "transform", parent=MMENGINE_TRANSFORMS, locations=["mmwave3d.datasets.transforms"]
)


MODELS = Registry("model", parent=MMENGINE_MODELS, locations=["mmwave3d.models"])

MODEL_WRAPPERS = Registry(
    "model_wrapper", parent=MMENGINE_MODEL_WRAPPERS, locations=["mmwave3d.models"]
)

WEIGHT_INITIALIZERS = Registry(
    "weight initializer",
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=["mmwave3d.models"],
)


OPTIMIZERS = Registry("optimizer", parent=MMENGINE_OPTIMIZERS)

OPTIM_WRAPPERS = Registry("optim wrapper", parent=MMENGINE_OPTIM_WRAPPERS)

OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    "optimizer wrapper constructor", parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
)

PARAM_SCHEDULERS = Registry("parameter scheduler", parent=MMENGINE_PARAM_SCHEDULERS)

METRICS = Registry("metric", parent=MMENGINE_METRICS, locations=["mmwave3d.evaluation"])

EVALUATOR = Registry(
    "evaluator", parent=MMENGINE_EVALUATOR, locations=["mmwave3d.evaluation"]
)


TASK_UTILS = Registry(
    "task util", parent=MMENGINE_TASK_UTILS, locations=["mmwave3d.models"]
)


VISUALIZERS = Registry("visualizer", parent=MMENGINE_VISUALIZERS)

VISBACKENDS = Registry("vis_backend", parent=MMENGINE_VISBACKENDS)


LOG_PROCESSORS = Registry("log_processor", parent=MMENGINE_LOG_PROCESSORS)
