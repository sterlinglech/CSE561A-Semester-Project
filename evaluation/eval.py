from rouge_score import rouge_scorer
import bert_score
from transformers import pipeline

class Evaluator:

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
        self.pipeline = pipeline("text-generation", model="gpt2")

    def rouge(self):
        precision_col = []
        recall_col = []
        fmeasure_col = []
        for i in range (len(self.dataframe)):
            scores = self.rouge_scorer.score(self.dataframe.loc[i]["human_summary"], self.dataframe.loc[i]["llm_summary"])["rougeLsum"]
            precision_col.append(scores.precision)
            recall_col.append(scores.recall)
            fmeasure_col.append(scores.fmeasure)
        self.dataframe.insert(len(self.dataframe.columns), "rouge_precision", precision_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "rouge_recall", recall_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "rouge_fmeasure", fmeasure_col, False)
        print(self.dataframe)

    def bert(self):
        human_summaries = list(self.dataframe["human_summary"])
        llm_summaries = list(self.dataframe["document"])
        precision_col, recall_col, fmeasure_col = bert_score.score(llm_summaries, human_summaries, lang="en")
        self.dataframe.insert(len(self.dataframe.columns), "bert_precision", precision_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "bert_recall", recall_col, False)
        self.dataframe.insert(len(self.dataframe.columns), "bert_fmeasure", fmeasure_col, False)
        print(self.dataframe)

    def llm(self):
        llm_summary = "the book was really good and it talked about making food"
        human_summary = "the book was bad and it was about baseball"
        prompt = f"Given this reference summary: {human_summary} Score this generated summary from 0 (poor) to 1 (perfect): {llm_summary} Respond with only the score and nothing else"
        response = self.pipeline(prompt, max_length=10, num_return_sequences=1)
        print(response[0])


    

    


