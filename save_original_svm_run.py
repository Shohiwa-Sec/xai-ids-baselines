"""Run original SVM training/evaluation with a test-column alignment fix.

Explanation sections are omitted and test columns aligned by name to training.
Model settings and feature values are unchanged. Saved inputs are NOT a raw-traffic pipeline.
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
SOURCE = ROOT / 'SVM_ALL_FINAL_original.py'
source = SOURCE.read_text(encoding='utf-8-sig')
marker = 'Y_train = Y_train.flatten()'
evaluation_marker = '# Assuming y_test and y_pred are your test labels and predicted labels respectively'
if source.count(marker) != 1 or source.count(evaluation_marker) != 1:
    raise RuntimeError('Original script changed; sections must be reviewed.')
training_end = source.index(marker)
evaluation_start = source.index(evaluation_marker)
evaluation_end = source.index('\npred_labels = np.argmax(y_pred, axis=1)', evaluation_start)
# Execute original training and evaluation statements, skipping intervening XAI.
# Blank lines preserve the original source line numbers for tracebacks.
prefix = (source[:training_end]
          + '\n' * source[training_end:evaluation_start].count('\n')
          + source[evaluation_start:evaluation_end])
# Match input columns by name before the original scaler and classifier calls.
# Keep this on the same source line so tracebacks retain original line numbers.
alignment_point = 'X_train_scaled = scaler.fit_transform(X_train)'
if prefix.count(alignment_point) != 1:
    raise RuntimeError('Original scaler section changed; review required.')
prefix = prefix.replace(alignment_point,
    "assert X_train.columns.is_unique and X_test.columns.is_unique; "
    "assert set(X_train.columns) == set(X_test.columns), 'Train/test feature sets differ'; "
    "X_test = X_test.loc[:, X_train.columns]; " + alignment_point)
ast.parse(prefix)

output = ROOT / 'results' / ('original_svm_saved_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
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
            joblib.dump(namespace['multi_target_clf'], output / 'svm_sgd.joblib')
            joblib.dump({key: namespace[key] for key in ['X_train', 'X_test', 'Y_train', 'Y_test']},
                        output / 'processed_inputs.joblib')
            joblib.dump({'actual': namespace['y_true_multiclass'], 'predicted': namespace['y_pred']},
                        output / 'evaluated_predictions.joblib')
            namespace['pd'].DataFrame(namespace['confusion'],
                index=namespace['class_names'], columns=namespace['class_names']).to_csv(output / 'confusion_matrix.csv')
            print(f'\nSaved SVM, processed inputs, predictions and output to:\n{output}', flush=True)
finally:
    os.chdir(previous_directory)

metadata = {
    'source': 'https://github.com/ogarreche/XAI_NIDS/blob/main/NSL-KDD/SVM_ALL_FINAL.py',
    'original_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'execution': 'Original training prefix plus original evaluation block; intervening XAI sections skipped.',
    'training_and_preprocessing_changes': 'Reorder test columns by name to match training before original scaler and classifier calls; feature values and all model settings unchanged.',
    'evaluated_model_parameters': namespace['multi_target_clf'].estimator.get_params(),
    'classes': namespace['class_names'],
    'python': platform.python_version(),
    'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
    'dataset_sha256': {name: hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest()
                       for name in ['KDDTrain+.txt', 'KDDTest+.txt']},
    'notes': [
        'Original SVM script uses SGDClassifier(loss=hinge), wrapped in MultiOutputClassifier; not SVC.',
        'Original SVM script does not time model training and prediction.',
        'Original minority oversampling, preprocessing and default SGD settings retained; estimator random_state is unset.',
        'Original separate test preprocessing, multioutput classification and hard-label AUC are preserved.',
        'Saved inputs are already processed. Do not feed raw traffic rows directly to saved models.',
        'Later XGEA input transformation and feature constraints still need to be agreed with the supervisor.',
    ],
}
(output / 'run_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
(output / 'source_snapshot.py').write_text(source, encoding='utf-8')
