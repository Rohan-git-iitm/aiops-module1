import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

df = mlflow.search_runs(
    experiment_names=["mnist-mlp"],
    order_by=["metrics.final_val_accuracy DESC"],
)

cols = [
    "tags.mlflow.runName",
    "params.lr",
    "params.batch_size",
    "metrics.final_val_accuracy",
    "metrics.train_loss",
    "run_id",
]
df[cols].to_csv("summary.csv", index=False)
print(df[cols].to_string(index=False))