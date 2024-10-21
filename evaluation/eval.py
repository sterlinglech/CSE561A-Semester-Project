from rouge_score import rouge_scorer
import bert_score
from transformers import pipeline
import openai
from dotenv import dotenv_values
from tqdm import tqdm

class Evaluator:

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
        openai.api_key = dotenv_values(".env")["OPENAI_API_KEY"]

    def rouge(self):
        precision_col = []
        recall_col = []
        fmeasure_col = []
        for i in range (len(self.dataframe)):
            scores = self.rouge_scorer.score(self.dataframe.loc[i]["reference_summary"], self.dataframe.loc[i]["generated_summary"])["rougeLsum"]
            precision_col.append(scores.precision)
            recall_col.append(scores.recall)
            fmeasure_col.append(scores.fmeasure)
        self.dataframe.insert(len(self.dataframe.columns), "rouge_precision", precision_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "rouge_recall", recall_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "rouge_fmeasure", fmeasure_col, False)
        print(self.dataframe)

    def bert(self):
        reference_summaries = list(self.dataframe["reference_summary"])
        generated_summaries = list(self.dataframe["generated_summary"])
        precision_col, recall_col, fmeasure_col = bert_score.score(generated_summaries, reference_summaries, lang="en")
        self.dataframe.insert(len(self.dataframe.columns), "bert_precision", precision_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "bert_recall", recall_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "bert_fmeasure", fmeasure_col, False)
        print(self.dataframe)

    def llm(self):
        precision_col = []
        recall_col = []
        overall_col = []
        precision_prompt = "Respond with a decimal score out to 3 digits from 0 (worst) to 1 (best) of how well the generated summary has information that is in the reference summary. If none of the information in the generated summary is in the reference summary respond with 0. If all of the information in the generated summary is in the reference summary respond with 1. Respond with only the score and nothing else.”"
        recall_prompt = "Respond with a decimal score out to 3 digits from 0 (worst) to 1 (best) of how well the generated summary covers all of the information in the reference summary. If all of the information in the reference summary is missing from the generated summary respond with 0. If all of the information in the reference summary is in the generated summary respond with 1. Respond with only the score and nothing else."
        overall_prompt = "Respond with a decimal score out to 3 digits from 0 (worst) to 1 (best) to score the generated summary based on the reference summary. If the summaries are not at all related respond with 0. If the summaries are semantically identical respond with 1. The score should also indicate how legally accurate and grammatically correct the generated summary is. Respond with only the score and nothing else."
        for i in tqdm(range(len(self.dataframe))):
            precision_response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Reference Summary: {self.dataframe.loc[i]['reference_summary']}"}, {"role": "user", "content": f"Generated Summary: {self.dataframe.loc[i]['generated_summary']}"}, {"role": "user", "content": precision_prompt}])
            recall_response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Reference Summary: {self.dataframe.loc[i]['reference_summary']}"}, {"role": "user", "content": f"Generated Summary: {self.dataframe.loc[i]['generated_summary']}"}, {"role": "user", "content": recall_prompt}])
            overall_response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Reference Summary: {self.dataframe.loc[i]['reference_summary']}"}, {"role": "user", "content": f"Generated Summary: {self.dataframe.loc[i]['generated_summary']}"}, {"role": "user", "content": overall_prompt}])
            precision_col.append(precision_response.choices[0].message.content)
            recall_col.append(recall_score = recall_response.choices[0].message.content)
            overall_col.append(overall_score = overall_response.choices[0].message.content)
        self.dataframe.insert(len(self.dataframe.columns), "llm_precision", precision_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "llm_recall", recall_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "llm_overall", overall_col, False)

    def dump(self, file):
        self.dataframe.to_csv(file, index=False) 

    

    


