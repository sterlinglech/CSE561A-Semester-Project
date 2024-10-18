from rouge_score import rouge_scorer
import bert_score
from transformers import pipeline
import openai
from dotenv import dotenv_values

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
        generated_summary = "During the search of the premises of the appellant No. 1 a complete working still was found which was being worked by the appellant No. 1 and his servant, appellant No. 2. The presidency Magistrate was satisfied that a working still and 516 illicit liquor were found. The appellant No. 1 was examined under section 342 of the Code of Criminal Procedure, he volunteered the statement that he did not know anything of the contraband seized by the police ; so no specific question about the still and other articles recovered from his premises were put by the Presidency Magistrate who convicted the appellants under sections 65(b), 65(f) & 66(b) of the Bombay Prohibition Act, relying on the facts of the recovery of still and illicit liquor and did not use the provision of section 103 for presumption against the appellants. The appellants on appeal by special leave contended, (1) that no presumption under section 103 of the Act could arise ; and that he had been denied the opportunity to rebut the presumption under section 103 of the Act, as no questions were put to them when they were examined under section 342 of the Code of Criminal Procedure (3) that as the Magistrate had not used the provision of section 103 for presumption against the appellants, the High Court ought not to have convicted the appellants on the presumption arising under section 103 of the Act without giving them an opportunity to rebut the same. On behalf of appellant No. 2 it was further urged that he was merely a servant of appellant No. 1; if any one was in possession of the still it was appellant No. 1 and no presumption against him could arise under section 103 of the Act. Held, that when an accused is examined under section 342 of the Code of Criminal Procedure and volunteers statement denying all knowledge of articles recovered from his possession, no prejudice is caused to him if no further questions are put to explain the possession of articles found in the premises occupied by him. The presumption which arises under section 103 of the Bombay Prohibition Act is that an offence under the Act is committed when a person is found in mere possession, without further evidence, of any still, utensil, implement or apparatus whatsoever for the manufacture of such intoxicant until contrary is proved. Thus no prejudice was caused to the appellant No. 1 when the High Court relied upon the presumption arising under section 103 of the Act to uphold his conviction under section 65(f) of the Act. Held, further, that it cannot be said of merely an employee in the premises that he was in physical possession of the things belonging to his master unless they were left in his custody, Where an offence under section 65(f) of the Bombay Prohibition Act has not been established beyond reasonable doubt and the possession of still does not amount to an offence under the section no presumption could arise under section 103 of the Act against a person that he was in possession of the still for which he could not account satisfactorily. In the instant case the still being in the possession of the master and there being no evidence that the employee in any 517 way aided his master to come into possession of the still, it could not be said that the appellant No. 2 was in such possession of the still as would amount to an offence under section 65(f) of the Act."
        reference_summary = "The appellant was the Ruler of the State of Baster which was later integrated with the State of Madhya Pradesh. He was recognised by the President as a Ruler under article 366(22) of the Constitution. The respondent resumed certain lands belonging to the appellant under the Madhya Pradesh Abolition of Proprietary Rights (Estates, Mahals, Alienated Lands) Act, 1950. The appellant contended that he was still a Ruler and not an ex Ruler and as such did not come within the definition of 'proprietor' given in the Act. Held, that the appellant was an ex Ruler for the purposes of the Act and was within the class of persons who were by name included in the definition of 'proprietor ' and was within the scope of the Act. Factually the appellant was an ex Ruler. He was a Ruler for the purposes of the privy purse guaranteed to him. There was nothing in article 366(22) which required a court to treat such a person as a Ruler for purposes outside the Constitution. Further, the appellant was also a maufidar in respect of the lands acquired which were exempt from the payment of rent or tax. The expression 'maufidar' was not necessarily confined to a grantee from a State or a Ruler of a State; he could be the holder of land which was exempted from payment of rent or tax."
        response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Reference Summary: {reference_summary}"}, {"role": "user", "content": f"Generated Summary: {generated_summary}"}, {"role": "user", "content": "Respond with a decimal score out to 3 digits from 0 (worst) to 1 (best) of how well the generated summary covers the reference summary. If the summaries are not at all related respond with 0. If the summaries have the same meaning respond with 1. Respond with only the score and nothing else."}])
        print(response.choices[0].message.content)

    

    


