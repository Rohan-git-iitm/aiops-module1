import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.MlflowClient()

runs = mlflow.search_runs(experiment_names=["mnist-mlp"])
frames = []
for _, r in runs.iterrows():
    rid = r["run_id"]
    loss = {m.step: m.value for m in client.get_metric_history(rid, "train_loss")}
    acc = {m.step: m.value for m in client.get_metric_history(rid, "val_accuracy")}
    d = pd.DataFrame({"epoch": sorted(loss)})
    d["run"] = r["tags.mlflow.runName"]
    d["train_loss"] = d["epoch"].map(loss)
    d["val_accuracy"] = d["epoch"].map(acc)
    frames.append(d)

pd.concat(frames).to_csv("curves.csv", index=False)
print("wrote curves.csv")