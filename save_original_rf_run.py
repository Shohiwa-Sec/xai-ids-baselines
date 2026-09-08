"""Run the original RF training/evaluation unchanged and save its artifacts.

Only the explanation section is omitted. No training settings or preprocessing
are changed. Saved processed inputs are NOT a reusable raw-traffic pipeline.
"""
from pathlib import Path
from datetime import datetime
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import sys

import joblib

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'RF_ALL_FINAL_original.py'
source = SOURCE.read_text(encoding='utf-8-sig')
marker = 'Y_train = Y_train.flatten()'
if source.count(marker) != 1:
    raise RuntimeError('Original script changed; stopping point must be reviewed.')
prefix = source.split(marker, 1)[0]
if "print('AUC_ROC total: '" not in prefix:
    raise RuntimeError('Evaluation section not found before stopping point.')
ast.parse(prefix)

output = ROOT / 'results' / ('original_rf_saved_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
output.mkdir(parents=True, exist_ok=False)


class Tee:
    def __init__(self, screen, file):
        self.screen, self.file = screen, file

    def write(self, text):
        self.screen.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.screen.flush()
        if not self.file.closed:
            self.file.flush()

    def close(self):
        # External logging may retain this stream until interpreter shutdown.
        # The context manager owns the log file; never close the real terminal.
        self.flush()


namespace = {'__name__': '__main__', '__file__': str(SOURCE)}
previous_directory = Path.cwd()
try:
    # Original code uses relative dataset paths.
    os.chdir(ROOT / 'data')
    with (output / 'console_output.txt').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print('Running original training and evaluation; explanations are skipped.', flush=True)
            exec(compile(prefix, str(SOURCE), 'exec'), namespace)
            # Preserve both configurations, not just the one evaluated last.
            joblib.dump(namespace['multi_target_rf'], output / 'first_rf_200_trees.joblib')
            joblib.dump(namespace['model_rf'], output / 'evaluated_rf_10_trees.joblib')
            joblib.dump({key: namespace[key] for key in ['X_train', 'X_test', 'Y_train', 'Y_test']},
                        output / 'processed_inputs.joblib')
            joblib.dump({'actual': namespace['y_true_multiclass'], 'predicted': namespace['preds']},
                        output / 'evaluated_predictions.joblib')
            namespace['pd'].DataFrame(namespace['confusion'],
                index=namespace['class_names'], columns=namespace['class_names']).to_csv(output / 'confusion_matrix.csv')
            print(f'\nSaved both trained models, processed inputs, predictions and output to:\n{output}', flush=True)
finally:
    os.chdir(previous_directory)

metadata = {
    'source': 'https://github.com/ogarreche/XAI_NIDS/blob/main/NSL-KDD/RF_ALL_FINAL.py',
    'original_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'execution': 'Original prefix through evaluation, before Y_train.flatten and explanation section.',
    'training_and_preprocessing_changes': 'None',
    'first_model_parameters': namespace['multi_target_rf'].estimator.get_params(),
    'evaluated_model_parameters': namespace['model_rf'].get_params(),
    'classes': namespace['class_names'],
    'python': platform.python_version(),
    'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
    'dataset_sha256': {name: hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest()
                       for name in ['KDDTrain+.txt', 'KDDTest+.txt']},
    'notes': [
        'Reported metrics evaluate the SECOND model (10 trees, depth 5).',
        'Printed training/prediction timings refer to the FIRST model (200 trees, depth 15).',
        'Second model has no fixed random seed; repeated results may differ.',
        'Original separate test preprocessing, multioutput classification and hard-label AUC are preserved.',
        'Saved inputs are already processed. Do not feed raw traffic rows directly to saved models.',
        'Later XGEA input transformation and feature constraints still need to be agreed with the supervisor.',
    ],
}
(output / 'run_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
(output / 'source_snapshot.py').write_text(source, encoding='utf-8')
