from rouge_score import rouge_scorer
import bert_score
from transformers import pipeline
import openai
from dotenv import dotenv_values
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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

    def bert(self):
        reference_summaries = list(self.dataframe["reference_summary"])
        generated_summaries = list(self.dataframe["generated_summary"])
        precision_col, recall_col, fmeasure_col = bert_score.score(generated_summaries, reference_summaries, lang="en")
        self.dataframe.insert(len(self.dataframe.columns), "bert_precision", precision_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "bert_recall", recall_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "bert_fmeasure", fmeasure_col, False)

    def llm(self):
        precision_col = []
        recall_col = []
        fmeasure_col = []
        overall_col = []
        precision_prompt = "Respond with a decimal score out to 3 digits from 0 (worst) to 1 (best) of how well the generated summary has information that is in the reference summary. If none of the information in the generated summary is in the reference summary respond with 0. If all of the information in the generated summary is in the reference summary respond with 1. Respond with only the score and nothing else.”"
        recall_prompt = "Respond with a decimal score out to 3 digits from 0 (worst) to 1 (best) of how well the generated summary covers all of the information in the reference summary. If all of the information in the reference summary is missing from the generated summary respond with 0. If all of the information in the reference summary is in the generated summary respond with 1. Respond with only the score and nothing else."
        overall_prompt = "Respond with a decimal score out to 3 digits from 0 (worst) to 1 (best) to score the generated summary based on the reference summary. If the summaries are not at all related respond with 0. If the summaries are semantically identical respond with 1. The score should also indicate how legally accurate and grammatically correct the generated summary is. Respond with only the score and nothing else."
        for i in tqdm(range(len(self.dataframe))):
            precision_response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Reference Summary: {self.dataframe.loc[i]['reference_summary']}"}, {"role": "user", "content": f"Generated Summary: {self.dataframe.loc[i]['generated_summary']}"}, {"role": "user", "content": precision_prompt}])
            recall_response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Reference Summary: {self.dataframe.loc[i]['reference_summary']}"}, {"role": "user", "content": f"Generated Summary: {self.dataframe.loc[i]['generated_summary']}"}, {"role": "user", "content": recall_prompt}])
            overall_response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Reference Summary: {self.dataframe.loc[i]['reference_summary']}"}, {"role": "user", "content": f"Generated Summary: {self.dataframe.loc[i]['generated_summary']}"}, {"role": "user", "content": overall_prompt}])
            precision_score = float(precision_response.choices[0].message.content)
            recall_score = float(recall_response.choices[0].message.content)
            fmeasure_score = 2 * ((precision_score * recall_score) / (precision_score + recall_score))
            overall_score = float(overall_response.choices[0].message.content)
            precision_col.append(precision_score)
            recall_col.append(recall_score)
            fmeasure_col.append(fmeasure_score)
            overall_col.append(overall_score)
        self.dataframe.insert(len(self.dataframe.columns), "llm_precision", precision_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "llm_recall", recall_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "llm_fmeasure", fmeasure_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "llm_overall", overall_col, False)

    def dump(self, file):
        self.dataframe.to_csv(file, index=False)

    def box_plot(self, cols=None):
        if cols == "rouge":
            numeric_columns = ["rouge_precision", "rouge_recall", "rouge_fmeasure"]
        elif cols == "bert":
            numeric_columns = ["bert_precision", "bert_recall", "bert_fmeasure"]
        elif cols == "llm":
            numeric_columns = ["llm_precision", "llm_recall", "llm_fmeasure", "llm_overall"]
        else:
            numeric_columns = self.dataframe.columns[2:]
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.dataframe[numeric_columns])
        plt.xticks(rotation=45)
        plt.title('Box and Whisker Plot for Evaluation Metrics')
        plt.show()

    def scatter(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.dataframe['rouge_fmeasure'], self.dataframe['llm_fmeasure'], color='blue', label='ROUGE vs LLM')
        plt.xlabel('ROUGE F-measure')
        plt.ylabel('LLM F-measure')
        plt.title('ROUGE F-measure vs LLM F-measure')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.figure(figsize=(8, 6))
        plt.scatter(self.dataframe['bert_fmeasure'], self.dataframe['llm_fmeasure'], color='green', label='BERT vs LLM')
        plt.xlabel('BERT F-measure')
        plt.ylabel('LLM F-measure')
        plt.title('BERT F-measure vs LLM F-measure')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.figure(figsize=(8, 6))
        plt.scatter(self.dataframe['rouge_fmeasure'], self.dataframe['bert_fmeasure'], color='red', label='ROUGE vs BERT')
        plt.xlabel('ROUGE F-measure')
        plt.ylabel('BERT F-measure')
        plt.title('ROUGE F-measure vs BERT F-measure')
        plt.grid(True)
        plt.legend()
        plt.show()



    

    


