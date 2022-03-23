import logging
import wikipedia as wiki
import re
from document_reader import DocumentReader
from keybert import KeyBERT

logger = logging.getLogger(__name__)


class WikiQA:
    """
    keywords extract question: what is the keywords, key phrases, meaning phrase, person name, entity name ?
    intro context: hello! I am glad to see you. I am a Pepper robot, an AI robot in JAIST.  I was born on 1 Jan 2022 at JAIST. I was born to support human search information on the Wikipedia Knowledge Base.  The persons who constructed me are researchers at JAIST, Phuong et al. I can support human search information on the Wikipedia Knowledge Base.  I love you. 
    """
    def __init__(self, model_name_or_path='deepset/roberta-base-squad2', using_kw_extract=False) -> None:
        self.reader = DocumentReader(model_name_or_path)
        self.using_kw_extract = using_kw_extract
        if using_kw_extract:
            self.kw_model = KeyBERT()

        # what is the keywords, key phrases, meaning phrase, person name, entity name ?

        # some other model
        # self.reader = DocumentReader("bert-large-uncased-whole-word-masking-finetuned-squad")
        # self.reader = DocumentReader("deepset/xlm-roberta-base-squad2")
        pass

    def answer(self, questions):
        results_reader = [""]
        intro_context = """
        hello! I am glad to see you. I am a Pepper robot, an AI robot in JAIST. My price is around 1800 $, but my benefit is unccountable.
        I was born on 1 Jan 2022 at JAIST. I was born to support human search information on the Wikipedia Knowledge Base. Researchers at JAIST created me with their love. I can support human search information on the Wikipedia Knowledge Base.  I love you. 
        """
        for question in questions:
            # simple preprocess question
            if not question.strip().endswith("?"):
                for check_wh_question in ["what", "when", "where", "why", "how", "who", "which"]:
                    if check_wh_question in question.lower():
                        question = question.strip() + "?"
            logger.info(f"Question: {question}")

            # wiki search page
            results_wiki_by_kw = []
            if self.using_kw_extract:
                # extract keywords
                keywords = self.kw_model.extract_keywords(question,
                                                          keyphrase_ngram_range=(1, 5), stop_words=None)

                logger.info(f"Keywords: {keywords}")
                logger.info(f"Searching by best keyword: {keywords[0][0]}")
                if keywords[0][0] != question:
                    results_wiki_by_kw = wiki.search(keywords[0][0], results=2)

            results_wiki = []
            if len(results_wiki_by_kw) == 0:
                results_wiki = wiki.search(question, results=2)

            text = ''
            for result in set(results_wiki + results_wiki_by_kw):
                try:
                    page = wiki.page(result)
                    logger.info(f"Add wiki page: {result}")
                    paras = re.split(r'\n\n+', page.content)
                    text = text + "\n\n\n" + "\n\n".join(paras[:10])
                except Exception:
                    logger.info(
                        "Can not get page {} from wiki DB".format(result))

            # ranking and generate answer by ML model
            self.reader.tokenize(question, text, intro_context=intro_context, max_toks=200)
            answer, detail_result = self.reader.get_answer()

            if len(answer.strip()) == 0 or "<s>" in answer.lower():
                answer = "Sorry, I do not find the information to answer of this question."

            if len(detail_result) == 0:
                continue

            context_found = (detail_result[0]['context'].replace(
                "<pad>", "").replace("[PAD]", "")).encode('utf-8').strip()
            logger.info(f"=======\n")
            logger.info(f"=======\nAnswer: {answer}")
            logger.info(f"=======\nConfidence: {detail_result[0]['prob']}")
            logger.info(f"=======\nContext: {context_found}")
            logger.info(f"=======\n")

        return answer, detail_result
