# Implementation design

## Requirements interpretation

The source document begins with a student who has no hardware and requests a dataset-and-virtual-simulation project. Its later examples switch to physical sensors. This implementation follows the opening scope while retaining the documented temperature, vibration and current inputs, Random Forest example, model persistence, Flask interface and failure/normal response.

The source also mentions LSTM, but its actual training example uses Random Forest. Random Forest is selected to make the first implementation understandable. No LSTM performance claim is made.

## Explicit implementation assumptions

The document does not supply a CSV, a label-generation method, sensor calibration or a future prediction horizon. This project therefore generates its own clearly identified fictional CSV and labels failures within the next ten observations. Sampling is once per simulated minute. Temperature is in Celsius, current in amps and vibration in arbitrary units. These choices are implementation decisions, not values sourced from the document or real machines.

The generated sensors gradually reflect an invented wear variable with noise. A model can learn the relationship between the resulting sensor levels and proximity to the generated terminal event. This demonstrates the pipeline, not validation of a scientific model of degradation.

## Modules

| File | Responsibility |
| --- | --- |
| main.py | Command routing, sequential full run and saved run log |
| src/config.py | Consistent project paths, fields, horizon and seed |
| src/data.py | Seeded synthetic histories and CSV loading |
| src/features.py | Current sensor values and three interaction features |
| src/train.py | Machine split, median imputer, model training, baselines, scores and graphs |
| src/predict.py | Model loading, API validation and shared prediction/action formatting |
| src/simulate.py | Held-out-machine replay, alerts, action records and timeline graph |
| src/api.py | Local dashboard, metrics, charts and /predict endpoint |
| src/dashboard.html | Browser form, evaluation display and generated charts |
| tests/test_project.py | Data integrity, leakage and interface checks |

## Data separation

A seeded permutation assigns whole machine IDs to 70% training, 15% validation and 15% test groups. For 80 machines, the groups contain 56, 12 and 12 IDs. There is no shared machine between groups. The median imputer is fitted on training features only and is saved inside the same pipeline as the model.

There are three original sensor features and three multiplication features. The model receives neither the timestamp nor any identifier, wear variable or failure column. It receives only present readings, so the API requires no sequence buffer. Although the source histories are time-stamped, this implementation is a current-observation classifier with a future-event target rather than a sequence model.

## Training and evaluation

The forest uses 120 trees, maximum depth 10, minimum leaf size 3, balanced class weights and seed 42. The decision threshold is fixed at 0.5 before evaluation. No hyperparameter search is performed. Validation is a separate diagnostic partition retained for later learning; neither it nor test data affects fitting or the threshold in this version.

The majority-class baseline shows why accuracy cannot be the only measure. Metrics include accuracy, precision, recall, F1, average precision and confusion matrices. Evaluation counts labeled readings, not independently observed failure events. Test results should remain a final check; repeated model selection against the reported test figures would invalidate that interpretation.

## Replay and actions

The simulator restricts selection to test machines. It computes model scores using the same feature function and fitted imputer as training and the API. A score at least 0.5 produces an alert and a virtual STOP action. Lower scores produce a virtual RUN action. There are no GPIO operations.

The replay displays the complete recorded history, including after virtual STOP actions. It therefore measures retrospective warnings, not the effects of maintenance. The true future-failure window appears in the graph as an evaluation reference only. Earlier alerts are false positives for the fixed ten-reading target.

## Handling missing and invalid data

Historical missing readings are imputed using training medians. Derived interactions involving missing sensors are also imputed by their training medians. Missingness is introduced independently at an approximate 1% rate for each sensor during data generation.

The interactive API requires all three sensor values. It rejects strings, booleans, nulls, nonfinite values, additional fields and numbers outside documented demo bounds. This API policy keeps hand-entered predictions explicit while the dataset workflow demonstrates historical imputation. Values of 0 are never substituted for failed sensors.

## Reproduction

Python 3.12.14 and exact installed package versions are recorded for this delivery. Seeds fix the generated histories, machine assignment and forest randomness. The fitted model includes the Scikit-learn version in metadata. Regenerating the model with the pinned dependencies avoids depending on compatibility of saved Python model objects across environments.

Plots, metrics and CSVs are generated locally. Package installation is the only network-dependent setup step. Dataset generation, training, replay, dashboard and API operate locally. The server listens on 127.0.0.1 with Flask debug mode disabled.
