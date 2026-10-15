# Guided project implementation

We will build and understand the dataset-and-virtual-simulation version requested at the beginning of your document. The delivered source files are ready to inspect and run; the sequence below lets you learn the project one phase at a time. All implementation data is generated locally. The only downloaded items required for setup are the Python libraries named in the project dependencies.

## Before starting

Extract the project ZIP if using the packaged copy. Work inside the `AI-Predictive-Maintenance-IoT` folder containing `main.py`. Keep each command in the same project folder. Python 3.12 is the tested interpreter family.

The overall workflow is:

```text
Generate sensor histories -> prepare features -> train model -> evaluate
-> replay unseen machine -> predict through API -> capture proof -> publish
```

The simulation represents a machine using three numbers: temperature, vibration and current. It generates a known fictional future failure, then learns which earlier sensor readings indicate a failure within the next ten observations. Model inputs do not include that future event.

## Phase 1 Environment setup

**What to do:** Open PowerShell in the project folder and check Python:

```powershell
py -3.12 --version
```

Then create an isolated environment and install the recorded dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe --version
```

If `py` is absent, check `python --version`. When it reports Python 3.12, use `python -m venv .venv` for environment creation.

On the original Codex computer, a Python 3.12.14 runtime was also available at the following path. If neither launcher is available, this command creates the same environment using that existing runtime:

```powershell
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
```

The included `setup.ps1` automates these setup choices when local script execution is available. Explicit environment commands avoid needing activation or changing PowerShell execution policy.

For Mac/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python --version
```

**Why:** Isolating dependencies prevents interference with other projects and makes the recorded package versions reproducible.

**Expected output:** A `.venv` folder, successful package installation and a Python 3.12 version message.

**Common mistake:** Installing with a different Python from the one used to run the project. Use the explicit environment interpreter for every command.

**Verify:** Run this on Windows:

```powershell
.\.venv\Scripts\python.exe -c "import pandas, numpy, sklearn, matplotlib, flask, joblib; print('Environment ready')"
```

The following phases show Windows commands. For Mac/Linux replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Phase 2 Generate and inspect the dataset

**What to do:**

```powershell
.\.venv\Scripts\python.exe main.py generate
.\.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_csv('data/iot_sensor_data.csv'); print(d.head().to_string(index=False)); print(d.shape); print(d.columns.tolist())"
```

Open `src/data.py` and `data/DATASET.md` alongside the output. The generator creates 80 fictional machines with time-stamped temperature, vibration and current readings. Default seed 42 produces 16,764 rows in the recorded environment.

**Why:** The supplied document expects a dataset but does not include one. Generating labelled data makes the simulation runnable without external sources or hardware.

**Expected output:** `data/iot_sensor_data.csv`, with machine identifiers, timestamps, three sensors, the actual failure event and a future-failure target.

**Common mistake:** Calling synthetic rows real sensor measurements. Describe them as fictional simulation data in your README and interviews.

**Verify:** Each machine has exactly one terminal `failure=1` event. The target `failure_next_10_readings=1` appears on its ten immediately preceding readings. The event row itself is excluded during training.

## Phase 3 Inspect cleaning and missing data

**What to do:**

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_csv('data/iot_sensor_data.csv'); print(d.isna().sum()); print(d[['temperature','vibration','current']].describe())"
```

Read the `SimpleImputer` pipeline step in `src/train.py`. Approximately 1% of values per sensor are intentionally missing. The pipeline learns training medians and uses them to fill missing sensor and interaction values. No separate command fits preprocessing before the split.

**Why:** Missing values represent a simple data-quality problem. Fitting preprocessing only on training machines prevents evaluation information from influencing the learned replacements.

**Expected output:** Missing-value counts and descriptive statistics. Missing entries in the original CSV are retained as evidence; the fitted pipeline handles them when needed.

**Common mistake:** Filling missing values using medians calculated from the entire dataset before splitting.

**Verify:** The project tests later compare saved imputer medians against training-only medians.

## Phase 4 Understand feature engineering and the target

**What to do:**

```powershell
.\.venv\Scripts\python.exe -c "from src.data import load_data; from src.features import make_features; print(make_features(load_data().head()).to_string(index=False))"
```

Read `src/features.py`. Six features are used: the three present sensor readings and three pairwise products. The products let the model consider combined sensor conditions. A missing sensor also makes its products missing; the same pipeline imputes them.

**Why:** Features are the information the model is allowed to use. The target is the answer it must learn to predict from that information.

**Expected output:** Six numeric feature columns, with possible missing entries before imputation. No machine ID, timestamp, failure event or future label appears in these features.

**Common mistake:** Including `failure` or `failure_next_10_readings` as inputs. That would reveal the answer rather than demonstrate prediction.

**Verify:** Change a label in a copied row and confirm its computed features stay unchanged. The automated feature test checks this property.

## Phase 5 Train and save the model

**What to do:**

```powershell
.\.venv\Scripts\python.exe main.py train
```

Read `src/train.py`. It assigns whole machines to train, validation and test groups, then fits median imputation and a Random Forest on the training group only. The decision threshold is fixed at 0.5. The implementation uses Random Forest consistently; it does not train the LSTM mentioned elsewhere in the document.

**Why:** The model learns from past labelled examples. Holding out entire machines tests the pipeline on different synthetic devices from those used in fitting.

**Expected output:** A `.pkl` file in `models/`, split IDs and metrics in `outputs/`, and evaluation charts in `images/`.

**Common mistake:** Randomly mixing readings from one machine across training and test data, or regenerating the data after training and assuming the saved model still corresponds to it.

**Verify:** There must be 56 training, 12 validation and 12 test IDs with no overlaps for the default run. Model persistence includes both preprocessing and the forest.

## Phase 6 Evaluate the results

**What to do:** Open `outputs/metrics.json`, `images/confusion_matrix.png` and `images/precision_recall.png`.

The recorded synthetic test run produced:

| Measure | Meaning | Recorded result |
| --- | --- | ---: |
| Precision | Correct positive predictions divided by all positive predictions | 84.56% |
| Recall | Detected positive readings divided by all true positive readings | 95.83% |
| F1 | Combined precision and recall measure | 89.84% |
| Accuracy | Correct predictions divided by all test readings | 98.91% |

The majority baseline reaches about 94.97% accuracy by predicting no imminent failures, while detecting none of the positive readings. That comparison explains why recall and precision matter.

**Why:** Evaluation separates a working program from a demonstrated ability to recognize the simulation's target.

**Expected output:** Confusion-matrix counts 2,243 true negatives, 21 false positives, 5 false negatives and 115 true positives for the recorded default run.

**Common mistake:** Describing high accuracy as industrial reliability, or interpreting reading-level recall as the percentage of machines whose failure was detected. These readings have overlapping future windows and are correlated.

**Verify:** Sum the four confusion-matrix cells to obtain 2,384 test readings. The positive row count is 120. Treat these as measured simulation outputs, not guaranteed outcomes for another dataset.

## Phase 7 Predict and replay a virtual machine

**What to do:**

```powershell
.\.venv\Scripts\python.exe main.py predict --temperature 35 --vibration 0.3 --current 0.6
.\.venv\Scripts\python.exe main.py predict --temperature 80 --vibration 4.5 --current 1.7
.\.venv\Scripts\python.exe main.py simulate
```

For slower playback:

```powershell
.\.venv\Scripts\python.exe main.py simulate --delay 0.05
```

Read `src/predict.py` and `src/simulate.py`. A score of at least 0.5 creates an alert and a simulated STOP action. Other scores create a simulated RUN action. The simulator chooses an unseen test machine, not one the model learned from.

**Why:** This connects the saved model to a virtual IoT workflow: receive a reading, predict, alert and record an action.

**Expected output:** Prediction JSON, `outputs/simulation_predictions.csv`, `outputs/simulation_summary.json` and `images/failure_prediction.png`.

**Common mistake:** Claiming a virtual STOP prevented failure. Replay continues along the pre-generated history so its actual consequences are not simulated. An alert earlier than the defined ten-reading window counts as a false positive for that target.

**Verify:** Check that the CSV contains a prediction, risk score and virtual motor action for each replayed reading. The default machine is SIM-036, with 244 replayed readings, 13 alert readings and 3 false alert readings in the recorded run.

## Phase 8 Open the dashboard and test the API

**What to do:**

```powershell
.\.venv\Scripts\python.exe main.py serve
```

Open `http://127.0.0.1:5000` in your browser. Enter the three sensor values and select **Predict risk**. Review the precision, recall, F1, confusion matrix, replay graph and sensor histories.

In a second PowerShell terminal, test the API directly:

```powershell
$body = @{temperature=80; vibration=4.5; current=1.7} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/predict' -Method Post -ContentType 'application/json' -Body $body
```

**Why:** The API provides a separate interface for predictions, and the dashboard makes the workflow easy to demonstrate.

**Expected output:** A JSON response containing `Prediction`, a risk score, the fixed threshold, horizon and virtual action. The dashboard displays that result alongside saved charts. Graphs are artifacts of the saved run, not live industrial telemetry.

**Common mistake:** Closing the server terminal before opening the page. If port 5000 is busy, start with `main.py serve --port 5001` and use the corresponding browser address.

**Verify:** Submit both sample sensor conditions. The dashboard and CLI should agree because they share the predictor. The API rejects malformed or incomplete input with an error response.

## Phase 9 Prepare your GitHub proof

**What to do:** Follow `docs/GITHUB_PROOF_PLAN.md`. Read and personalize the README, retain the synthetic-data disclosure, capture your own screenshots and create the repository when ready.

**Why:** Source files, execution evidence, evaluation artifacts and an understandable README make the work reviewable.

**Expected output:** A repository containing your implementation and evidence, with a description explaining that this is a dataset-based simulation.

**Common mistake:** Uploading `.venv`, claiming hardware was used, inventing commit dates or presenting fictional sensor data as industrial data.

**Verify:** A new checkout should be installable from `requirements.txt`, and `main.py run` should recreate the artifacts. GitHub publication has not been performed for this delivery.

## Phase 10 Verify and demonstrate the complete project

**What to do:**

```powershell
.\.venv\Scripts\python.exe main.py run
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Why:** A final full run confirms that generation, model fitting, prediction, visualization and replay work together. Integrity tests verify the most consequential mistakes this kind of project can make.

**Expected output:** A complete run log, regenerated artifacts and seven passing tests.

**Common mistake:** Repeatedly adjusting the model to maximize the reported test scores. If you later experiment, make choices against validation data and preserve an untouched test set for final evaluation.

**Verify:** Explain the three inputs, the future target, the machine split, the confusion matrix and the virtual motor action in your own words. Be explicit about which assumptions were invented and which requirements came from the document.

## A short demonstration sequence

1. State that the project uses locally generated synthetic sensor data without hardware.
2. Show five CSV rows and explain the target.
3. Show the training command, metrics and confusion matrix.
4. Run the unseen-machine replay and explain the alerts and false positives.
5. Open the dashboard, submit two sensor readings and show the JSON API response.
6. Show the source modules, passing integrity tests and README.

You can now explain and reproduce the project without claiming it has been tested on industrial machinery.
