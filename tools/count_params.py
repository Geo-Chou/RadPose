import argparse
from typing import Tuple

from mmengine import DefaultScope
from mmengine.config import Config, DictAction
from mmengine.registry import MODELS
from mmengine.utils import import_modules_from_strings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Count the number of parameters in a model config.')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    return parser.parse_args()


def import_custom_modules(cfg: Config) -> None:
    custom_imports = cfg.get('custom_imports', None)
    if not custom_imports:
        return
    imports = custom_imports.get('imports', [])
    allow_failed = custom_imports.get('allow_failed_imports', False)
    if isinstance(imports, str):
        imports = [imports]
    import_modules_from_strings(imports, allow_failed)


def set_default_scope(cfg: Config) -> None:
    scope = cfg.get('default_scope', None)
    if scope is None:
        return
    current_scope = DefaultScope.get_current_instance()
    if current_scope is None or current_scope.scope_name != scope:
        DefaultScope.get_instance(f'param_count_{scope}', scope_name=scope)


def count_parameters(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_count(num: int) -> str:
    return f'{num:,} ({num / 1e6:.2f} M)'


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_custom_modules(cfg)
    set_default_scope(cfg)

    model = MODELS.build(cfg.model)
    model.eval()

    total, trainable = count_parameters(model)
    frozen = total - trainable
    print(f'Config: {args.config}')
    print(f'Total parameters:     {format_count(total)}')
    print(f'Trainable parameters: {format_count(trainable)}')
    print(f'Frozen parameters:    {format_count(frozen)}')


if __name__ == '__main__':
    main()
