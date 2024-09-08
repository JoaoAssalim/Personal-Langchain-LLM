from sentence_transformers import SentenceTransformer, util


# https://sbert.net/
class Embed:
    """
    This Class is specific to get the score between the 
    user quest and the LLM response
    """
    def __init__(self, quest, answer):
        self.quest = quest
        self.answer = answer
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def convert_to_embed(self):
        query_embed = self.model.encode(self.quest, convert_to_tensor=True)
        answer_embed = self.model.encode(self.answer, convert_to_tensor=True)

        return query_embed, answer_embed

    def get_score_from_embeds(self):
        query_embed, answer_embed = self.convert_to_embed()
        relevance_score = util.cos_sim(query_embed, answer_embed).item()

        return relevance_score
