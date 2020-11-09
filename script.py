from src.experiment import run_experiment

params = [
    [('simple_fc', 1e-3, 1), {}],         # simple_fc(LR, EPOCHS), {default data}
    [('simple_fc2', 1e-4, 1e-5, 1), {}],  # simple_fc2(LR, WD, EPOCHS), {default data}
]

reports_fname = run_experiment(params)

print(reports_fname)