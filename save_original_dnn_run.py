"""Run original DNN training/evaluation with a documented syntax repair and save its artifacts.

Explanation sections are omitted and two commented normalization declarations restored.
Original architecture and both 50-epoch training calls are preserved. Saved processed inputs are NOT a reusable raw-traffic pipeline.
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
SOURCE = ROOT / 'DNN_ALL_FINAL_original.py'
source = SOURCE.read_text(encoding='utf-8-sig')
# Restore the two commented declarations whose function body remains active.
# Original source on disk stays unchanged; no architecture/settings are changed.
fixed_source = source
for old, new in [('#std_scaler = StandardScaler()', 'std_scaler = StandardScaler()'),
                 ('#def standardization(df,col):', 'def standardization(df,col):')]:
    if fixed_source.splitlines().count(old) != 1:
        raise RuntimeError('Original normalization block changed; review required.')
    fixed_source = fixed_source.replace(old, new)
shap_start = fixed_source.index('background_dataset = shap.sample(X_train, 500)')
fit_statement = 'model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0)'
if fixed_source.count(fit_statement) != 2:
    raise RuntimeError('Expected two original training calls; review required.')
second_fit = fixed_source.index(fit_statement, fixed_source.index(fit_statement) + 1)
evaluation_end = fixed_source.index('# In[31]:', second_fit)
prefix = (fixed_source[:shap_start]
          + '\n' * fixed_source[shap_start:second_fit].count('\n')
          + fixed_source[second_fit:evaluation_end])
ast.parse(prefix)

output = ROOT / 'results' / ('original_dnn_saved_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
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
            # Save the original fitted model without changing its training.
            namespace['model'].save(output / 'dnn.keras')
            joblib.dump({key: namespace[key] for key in ['X_train', 'X_test', 'Y_train', 'Y_test']},
                        output / 'processed_inputs.joblib')
            joblib.dump({'actual': namespace['y_true_multiclass'], 'predicted': namespace['preds']},
                        output / 'evaluated_predictions.joblib')
            namespace['pd'].DataFrame(namespace['confusion'],
                index=namespace['class_names'], columns=namespace['class_names']).to_csv(output / 'confusion_matrix.csv')
            print(f'\nSaved DNN, processed inputs, predictions and output to:\n{output}', flush=True)
finally:
    os.chdir(previous_directory)

metadata = {
    'source': 'https://github.com/ogarreche/XAI_NIDS/blob/main/NSL-KDD/DNN_ALL_FINAL.py',
    'original_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'execution': 'Original two 50-epoch training calls and final evaluation; intervening SHAP skipped; two normalization declarations restored.',
    'training_and_preprocessing_changes': 'Restore commented scaler and function declarations. Both original 50-epoch fits retained. No other preprocessing or training changes.',
    'evaluated_model_parameters': namespace['model'].get_config(),
    'classes': namespace['class_names'],
    'python': platform.python_version(),
    'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'tensorflow', 'keras']},
    'dataset_sha256': {name: hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest()
                       for name in ['KDDTrain+.txt', 'KDDTest+.txt']},
    'notes': [
        'Original DNN architecture: Dense 128 ReLU, Dense 64 ReLU, Dense 5 softmax; Adam, categorical crossentropy, batch size 32; two 50-epoch fits.',
        'Printed training time covers ONLY the first 50 epochs, not the subsequent 50-epoch continuation.',
        'No oversampling added. Original preprocessing and test-column order retained; neural-network seed is unset.',
        'Original separate test preprocessing, multioutput classification and hard-label AUC are preserved.',
        'Saved inputs are already processed. Do not feed raw traffic rows directly to saved models.',
        'Later XGEA input transformation and feature constraints still need to be agreed with the supervisor.',
    ],
}
(output / 'run_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
(output / 'source_snapshot.py').write_text(source, encoding='utf-8')
