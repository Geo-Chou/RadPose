from mmpose.apis import MMPoseInferencer

MODEL = 'rtmpose-m_8xb64-210e_mpii-256x256'


def get_meta_from_obj(obj):
    if obj is None:
        return None
    if hasattr(obj, 'dataset_meta') and getattr(obj, 'dataset_meta'):
        return obj.dataset_meta
    if hasattr(obj, 'metainfo') and getattr(obj, 'metainfo'):
        return obj.metainfo
    if hasattr(obj, 'data_preprocessor') and hasattr(obj.data_preprocessor, 'dataset_meta'):
        return obj.data_preprocessor.dataset_meta
    return None


def find_meta(inferencer):
    # direct meta on inferencer
    meta = get_meta_from_obj(inferencer)
    if meta:
        return meta

    # common attribute names across versions
    candidates = [
        'model', '_model', 'pose_model', '_pose_model',
        'inferencer', '_inferencer', 'pose2d_inferencer', '_pose2d_inferencer',
        'inferencers', '_inferencers'
    ]

    for name in candidates:
        if not hasattr(inferencer, name):
            continue
        obj = getattr(inferencer, name)
        if obj is None:
            continue
        if isinstance(obj, dict):
            for v in obj.values():
                meta = get_meta_from_obj(v)
                if meta:
                    return meta
                if hasattr(v, 'model'):
                    meta = get_meta_from_obj(v.model)
                    if meta:
                        return meta
        else:
            meta = get_meta_from_obj(obj)
            if meta:
                return meta
            if hasattr(obj, 'model'):
                meta = get_meta_from_obj(obj.model)
                if meta:
                    return meta

    # last resort: scan attributes shallowly
    for name in dir(inferencer):
        if not name.startswith('_') and ('model' in name.lower() or 'infer' in name.lower()):
            try:
                obj = getattr(inferencer, name)
            except Exception:
                continue
            meta = get_meta_from_obj(obj)
            if meta:
                return meta
            if hasattr(obj, 'model'):
                meta = get_meta_from_obj(obj.model)
                if meta:
                    return meta
    return None


def main():
    inferencer = MMPoseInferencer(pose2d=MODEL, show_progress=False)
    meta = find_meta(inferencer)
    if not meta:
        print('No dataset meta found on inferencer/model')
        return

    keypoint_id2name = meta.get('keypoint_id2name')
    keypoint_name = meta.get('keypoint_name')

    if isinstance(keypoint_id2name, dict):
        for idx in sorted(keypoint_id2name.keys()):
            print(f"{idx}: {keypoint_id2name[idx]}")
        return

    if isinstance(keypoint_name, (list, tuple)):
        for idx, name in enumerate(keypoint_name):
            print(f"{idx}: {name}")
        return

    print('No keypoint name list found in meta')


if __name__ == '__main__':
    main()
