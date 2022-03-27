import logging
import re
from document_reader import DocumentReader
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder, DprQueryEncoder
from pyserini.search.hybrid import HybridSearcher
from utils_qa import _norm_text_ascii

from wiki_qa import WikiQA
import json

logger = logging.getLogger(__name__)


class WikiQALucene(WikiQA):
    """
    keywords extract question: what is the keywords, key phrases, meaning phrase, person name, entity name ?
    intro context: hello! I am glad to see you. I am a Pepper robot, an AI robot in JAIST.  I was born on 1 Jan 2022 at JAIST. I was born to support human search information on the Wikipedia Knowledge Base.  The persons who constructed me are researchers at JAIST, Phuong et al. I can support human search information on the Wikipedia Knowledge Base.  I love you. 
    """
    def __init__(self, use_wiki=False, use_msmarco=True, **kwargs) -> None:
        super().__init__(**kwargs)
        if use_msmarco:
            # passage get from Bing-search engine data 
            self.ssearcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage') #facebook-dpr-question_encoder-single-nq-base
            encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
            dsearcher = FaissSearcher.from_prebuilt_index(
                'msmarco-passage-tct_colbert-hnsw',
                encoder
            )
            self.hsearcher = HybridSearcher(dsearcher, self.ssearcher)
            # self.hsearcher = self.ssearcher

        elif use_wiki:
            self.ssearcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr') #facebook-dpr-question_encoder-single-nq-base
            encoder = DprQueryEncoder('facebook/dpr-ctx_encoder-single-nq-base')
            dsearcher = FaissSearcher.from_prebuilt_index(
                'wikipedia-dpr-multi-bf',
                encoder
            )
            self.hsearcher = HybridSearcher(dsearcher, self.ssearcher)
            # self.hsearcher = self.ssearcher

    def answer(self, questions):
        results_reader = [""]
        intro_context = """
        hello! I am glad to see you. I am a Pepper robot, an AI robot in JAIST. My price is around 1800 $, but my benefit is unccountable.
        I was born on 1 Jan 2022 at JAIST. I was born to support human search information on the Wikipedia Knowledge Base. Researchers at JAIST created me with their love. I can support human search information on the Wikipedia Knowledge Base.  I love you. 
        """
        for question in questions:
            question = self.text_norm(question)

            # check the caced question 
            question_or_ans = self.refine_question_by_cached(question)
            if isinstance(question_or_ans, tuple):
                return question_or_ans
            else:
                question = question_or_ans

            # simple preprocess question
            if not question.strip().endswith("?"):
                for check_wh_question in ["what", "when", "where", "why", "how", "who", "which"]:
                    if check_wh_question in question.lower():
                        question = question.strip() + "?"
            logger.info(f"Question: {_norm_text_ascii(question)}")

            # wiki search page
            results = self.hsearcher.search(question,  k=15)
             
            text = ''
            for result in results[:15]:
                try:
                    doc = self.ssearcher.doc(result.docid)
                    json_doc = json.loads(doc.raw()).get('contents', '').encode('ascii', 'ignore').decode('ascii')
                    logger.info(f"Add wiki page: {result.docid}: {_norm_text_ascii(json_doc[:40])}")

                    text = text + "\n\n\n" +  json_doc
                except Exception as e:
                    logger.info(e)
                    logger.info(
                        "Can not get page {} from External DB".format(result))

            # ranking and generate answer by ML model
            self.reader.tokenize(question, text, intro_context=intro_context, max_toks=500)
            answer, detail_result = self.reader.get_answer()

            logger.info(f'len(detail_result) = {len(detail_result)}')
            if len(answer.strip()) == 0 or "<s>" in answer.lower():
                answer = "Sorry, I do not find the information to answer of this question."

            if len(detail_result) == 0:
                continue

            context_found = (detail_result[0]['context'].replace(
                "<pad>", "").replace("[PAD]", "")).encode('utf-8').strip()
            logger.info(f"=======\n")
            logger.info(f"=======\nAnswer: {_norm_text_ascii(answer)}")
            logger.info(f"=======\nConfidence: {detail_result[0]['prob']}")
            logger.info(f"=======\nContext: {context_found}")
            logger.info(f"=======\n")

        return answer, detail_result
