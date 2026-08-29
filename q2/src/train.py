import argparse
import warnings

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_mnist(n_train, n_val, seed):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    return train_test_split(
        X, y, train_size=n_train, test_size=n_val, random_state=seed, stratify=y
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mnist-mlp")

    X_train, X_val, y_train, y_val = load_mnist(args.n_train, args.n_val, args.seed)

    run_name = f"mlp-lr{args.lr}-bs{args.batch_size}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("hidden_size", args.hidden_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("n_train", args.n_train)
        mlflow.log_param("seed", args.seed)

        model = MLPClassifier(
            hidden_layer_sizes=(args.hidden_size,),
            learning_rate_init=args.lr,
            batch_size=args.batch_size,
            max_iter=1,
            warm_start=True,
            random_state=args.seed,
        )

        for epoch in range(args.epochs):
            model.fit(X_train, y_train)
            train_loss = model.loss_
            val_acc = accuracy_score(y_val, model.predict(X_val))
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        mlflow.log_metric("final_val_accuracy", val_acc)
        mlflow.sklearn.log_model(model, 
                                 name="model",
                                 skops_trusted_types=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"]
                                 )

        print(f"{run_name}  train_loss={train_loss:.4f}  val_accuracy={val_acc:.4f}")


if __name__ == "__main__":
    main()