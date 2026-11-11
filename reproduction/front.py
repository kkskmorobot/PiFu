"""Start the preserved frontend with the portable dataset and exact selected models."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import bootstrap


def build_app():
    data=bootstrap.configure(require_data=True)
    root=bootstrap.PROJECT_ROOT
    spec=importlib.util.spec_from_file_location('portable_dermaseg',root/'QianDuan/app.py')
    app=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    index=json.loads((root/'reproduction/frozen_models.json').read_text(encoding='utf-8'))['models']
    def discover_models():
        models=[]
        # LG3 is the thesis interface example; LGX remains the predeclared seed43 representative.
        for run in sorted(index,key=lambda r:(r!='lg3',r!='lgx',r)):
            item=index[run]
            numeric=int(run[3:]) if run.startswith('run') else None
            models.append({'id':run,'run':numeric if numeric is not None else run,
                           'run_number':numeric,'series':str(numeric) if numeric is not None else run,
                           'tool_dir':run,'label':f'{run.upper()} · frozen FP32 EMA · epoch {item["visible_epoch"]}',
                           'experiment':item['experiment'],'weights':root/item['weights'],
                           'family':'YOLO26 二值分割 + 八分类','class_count':8,'trained_at':0})
        return models
    app.discover_models=discover_models
    app.SAMPLE_DATA_ROOT=data/'processed_skin7_v3'
    app.SAMPLE_IMAGES_ROOT=app.SAMPLE_DATA_ROOT/'images'
    return app


def main():
    build_app().main()


if __name__=='__main__':
    main()
