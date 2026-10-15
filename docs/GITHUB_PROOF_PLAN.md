# GitHub publishing and proof plan

This plan follows the source document's request for a professional repository, learning commits and visual evidence. The implementation has been run locally. No GitHub account has been accessed and no remote repository has been created or published in this delivery.

## Repository details

Suggested name: `AI-Predictive-Maintenance-IoT`

Suggested description: `Predictive maintenance student project using synthetic IoT sensor histories, Random Forest, a Flask API, and virtual failure-alert replay.`

Suggested topics: `python`, `predictive-maintenance`, `iot`, `machine-learning`, `random-forest`, `flask`, `synthetic-data`, `student-project`.

Personalize the README to reflect your understanding and accurately describe any help used in building the project. Keep the dataset assumptions and simulation limits visible.

## Prepare and publish

Create an empty repository in your GitHub account with the suggested name. Do not add another README there when publishing this existing folder. Choose repository visibility and a suitable license before public distribution.

From the project folder:

```text
git init
git branch -M main
git status
git add .
git commit -m "Build synthetic IoT predictive maintenance demonstration"
```

Check that `.venv`, `__pycache__` and `.cache` are excluded. Keep the small generated dataset, model, charts, metrics and proof logs so a reviewer can inspect them. The model can also be recreated by running the pipeline.

Replace `YOUR_USERNAME` in the next commands with your account name and use the repository you created:

```text
git remote add origin https://github.com/YOUR_USERNAME/AI-Predictive-Maintenance-IoT.git
git push -u origin main
```

The remote URL is a template, not an existing repository or a source for project content. Authentication may be required by your Git setup.

## Seven day learning and commit sequence

The delivered code already exists. Use the sequence below to learn, reproduce and make genuine improvements. Commit work when you perform it; do not invent past work or backdate commits. One honest initial commit followed by your own learning changes is also appropriate.

| Day | Activity | Suggested commit for actual work | Evidence |
| --- | --- | --- | --- |
| 1 | Set up Python environment, read the architecture, personalize overview | `Document project setup and scope` | Python version and successful import check |
| 2 | Generate data, inspect fields, explain label timing in your own words | `Explain synthetic sensor data and future labels` | Dataset preview and data dictionary |
| 3 | Understand missing values and feature products; add meaningful exploration notes | `Document preprocessing and feature checks` | Missing-data counts and feature preview |
| 4 | Train and explain the machine-separated split | `Record model training and machine split` | Training log and split IDs |
| 5 | Interpret confusion matrix, false alerts and baseline comparison | `Analyze synthetic model evaluation` | Metrics and your interpretation |
| 6 | Run replay and dashboard; capture the actual demonstration | `Add virtual replay and dashboard evidence` | Dashboard screenshots and prediction JSON |
| 7 | Review README, rerun checks and publish when ready | `Complete reproduction guide and proof checklist` | Passing tests and repository preview |

Avoid empty commits or many artificial commits made only to appear active. Explain changes in terms of what you implemented or learned.

## Proof checklist

The generated PNG files are graphs, not screenshots of a running application. Use the names below for screenshots you actually capture. Put those screenshots in `images/proof/`, then add them to your README.

| Evidence | Existing artifact or capture action | Suggested filename |
| --- | --- | --- |
| Dataset preview | Capture first rows of generated CSV with column headers | `images/proof/01_dataset_preview.png` |
| Missing-data analysis | Capture the Phase 3 terminal command and output | `images/proof/02_missing_data.png` |
| Training output | `outputs/run.log` plus an actual terminal screenshot | `images/proof/03_training_log.png` |
| Model evaluation | Existing generated graph | `images/confusion_matrix.png` |
| Precision and recall | Existing generated graph | `images/precision_recall.png` |
| Sensor histories | Existing generated graph | `images/sensor_history.png` |
| Failure replay | Existing generated graph and prediction CSV | `images/failure_prediction.png` |
| Normal prediction | Capture actual CLI or API JSON output | `images/proof/04_normal_prediction.png` |
| Failure alert | Capture actual CLI or API JSON output | `images/proof/05_failure_alert.png` |
| Dashboard | Capture the running local page after making a prediction | `images/proof/06_dashboard.png` |
| Tests | Capture actual test names and OK result | `images/proof/07_tests.png` |
| Repository preview | Capture your own repository after publishing | `images/proof/08_github_repository.png` |

For example, after capturing a dashboard screenshot, add this relative Markdown link in the README:

```markdown
![Local predictive maintenance dashboard](images/proof/06_dashboard.png)
```

Do not add placeholder screenshots or imply that the files in `images/proof/` exist before you capture them. The implementation already includes actual generated charts, a run log and metrics; those are valid existing proof artifacts.

## Interview explanation

Explain the project using these questions:

1. What does a positive label mean, and how far into the future does it look?
2. Which values are invented, and why was simulation used?
3. Why are whole machines kept separate across train and test sets?
4. Why can a model with high accuracy still miss every failure window?
5. What do the false positives and false negatives mean in this project?
6. How do the CLI, replay and API use the same model features?
7. What exactly does the virtual motor action demonstrate?
8. What additional evidence would be required before making claims about real machines?

Support answers with the code and actual generated outputs. The project demonstrates a complete software workflow under documented simulation assumptions.
