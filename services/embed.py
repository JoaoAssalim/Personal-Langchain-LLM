from sentence_transformers import SentenceTransformer, util
import numpy as np


# https://sbert.net/
class Embed:
    """
    This Class is specific to get the score between the 
    user quest and the LLM response
    """
    def __init__(self, quest, answer):
        self.quest = quest.lower().strip()
        self.answer = answer.lower().strip()
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def convert_to_embed(self):
        """
        This functions is to embed the answer and quest
        """
        query_embed = self.model.encode(self.quest, convert_to_tensor=True)
        answer_embed = self.model.encode(self.answer, convert_to_tensor=True)

        return query_embed, answer_embed

    def get_score_from_embeds(self):
        """
        This functions is to calculate the score of the bot answer.
        To do it, we use the combination of cosine similiarity and euclidian distance
        and after get each value of distance, we sum and divide the result by 2
        """
        query_embed, answer_embed = self.convert_to_embed()
        cos_sim_score = util.cos_sim(query_embed, answer_embed).item()
        euclidean_distance = np.linalg.norm(query_embed.cpu().numpy() - answer_embed.cpu().numpy())
        normalized_euclidean = 1 - (euclidean_distance / np.sqrt(len(query_embed)))
        combined_score = (cos_sim_score + normalized_euclidean) / 2

        return combined_score
