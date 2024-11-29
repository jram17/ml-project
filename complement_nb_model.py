import pandas as pd
from ComplementNB import ComplementNB

model = ComplementNB.load_model("complement_nb_model.pkl")

while True:
    inp = input("\nsentence: ")
    print(model.sentiment[model.predict(inp)])