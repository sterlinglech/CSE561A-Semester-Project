from datasets import load_dataset
import pandas as pd
from eval import Evaluator
from predict import BaseBartModel
from tqdm import tqdm


def main():

    # results = []
    # ds9_test = load_dataset("AjayMukundS/Legal_Text_Summarization-llama2", split='test')
    # base_model = BaseBartModel()

    # for i in tqdm(range(len(ds9_test))):
    #     generated_summary = base_model.predict(ds9_test[i]["text"])
    #     reference_summary = ds9_test[i]["summary"]
    #     results.append({'reference_summary': reference_summary, 'generated_summary': generated_summary})
    # df = pd.DataFrame(results)
    # df.to_csv('trained_results.csv', index=False)


    testing_data = pd.read_csv("trained_evals.csv")
    # print(testing_data.head())
    # testing_data = testing_data.drop("dataset_name", axis=1).rename({"judgement" : "document", "summary" : "human_summary"}, axis=1)
    # dtu = testing_data.loc[0:3]
    evaluator = Evaluator(testing_data)
    # evaluator.rouge()
    # print("done rouge")
    # evaluator.bert()
    # print("done bert")
    # evaluator.llm()
    # print("done llm")
    # evaluator.dump("trained_evals.csv")
    # evaluator.box_plot()
    evaluator.scatter()




 

    print("done")
    return




if __name__ == "__main__":
    main()