from datasets import load_dataset
import pandas as pd
from eval import Evaluator



def main():


    testing_data = pd.read_csv("data.csv")
    # print(testing_data.head())
    # testing_data = testing_data.drop("dataset_name", axis=1).rename({"judgement" : "document", "summary" : "human_summary"}, axis=1)
    # dtu = testing_data.loc[0:3]
    evaluator = Evaluator(testing_data)
    evaluator.rouge()
    evaluator.bert()
    evaluator.llm()
    evaluator.dump("evals.csv")
    # evaluator.box_plot()
    evaluator.scatter()


 

    print("done")
    return




if __name__ == "__main__":
    main()