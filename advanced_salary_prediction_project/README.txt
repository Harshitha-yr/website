# Advanced Web-Based Salary Prediction System

## Files
- `salary_data.csv`: synthetic demonstration dataset; replace with verified data for meaningful use.
- `train_model.py`: trains Linear Regression, Decision Tree, and Random Forest; saves Random Forest and comparison metrics.
- `app.py`: Streamlit interface with PDF report, model comparison, prediction history, and job role selection.
- `requirements.txt`: Python dependencies.

## Setup (VS Code terminal, in this folder)
```bash
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

## Dataset columns
`Experience,Education,Age,Skills,JobRole,Salary`

If your original `salary_data.csv` does not have `JobRole`, the training script adds `Software Engineer` to every record for compatibility. That is only a placeholder and does NOT teach the model real job-role effects. For meaningful role-based predictions, collect reliable salary records for multiple job roles.

The generated demo data is synthetic. Do not present its predictions or metrics as real-world salary accuracy.
