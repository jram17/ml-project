import pandas as pd
import matplotlib.pyplot as plt
from ComplementNB import ComplementNB

model = ComplementNB.load_model("complement_nb_model.pkl")

def show_metrics_graph(metrics_file):
    metrics_df = ComplementNB.load_metrics(metrics_file)
    print("Evaluation Metrics:\n")
    print(metrics_df)
    
    metrics_df[['precision', 'recall', 'f1_score']].plot.bar(figsize=(10, 6))
    plt.title("Evaluation Metrics by Class")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.legend(title="Metrics")
    plt.tight_layout()
    plt.show()

while True:
    print("\nOptions:")
    print("1. Predict sentiment of a sentence")
    print("2. Show evaluation metrics graph")
    print("3. Exit")

    choice = input("\nEnter your choice: ")
    
    if choice == "1":
        inp = input("\nEnter a sentence: ")
        print(f"Predicted sentiment: {model.sentiment[model.predict(inp)]}")
    elif choice == "2":
        show_metrics_graph("cnb_evaluation_metrics.pkl")
    elif choice == "3":
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please try again.")
