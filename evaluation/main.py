from datasets import load_dataset
import pandas as pd
from eval import Evaluator



def main():
    dataset = load_dataset("joelniklaus/legal_case_document_summarization") #7773 train, 200 test

    # training_data = pd.DataFrame(dataset["train"])
    testing_data = pd.DataFrame(dataset["test"])
    testing_data = testing_data.drop("dataset_name", axis=1).rename({"judgement" : "document", "summary" : "human_summary"}, axis=1)

    dtu = testing_data.loc[0:3]
    evaluator = Evaluator(dtu)
    evaluator.llm()


    print("done")
    return




if __name__ == "__main__":
    main()