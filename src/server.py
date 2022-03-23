from flask import request, jsonify, Flask, render_template
import logging
import json

from flask.wrappers import Response
from wiki_qa import WikiQA
from wiki_qa_lucene import WikiQALucene

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
PORT = 7009
HOST = "0.0.0.0"
qa_engine = None

trained_model = dict()
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, filename='./server_run.log',
                        filemode='a',)

 
@app.route('/', methods=['GET', 'POST'])
def answer_form():
    if request.method == "GET":
        return render_template('greeting.html')
    question = request.form['question']
    logger.info(question)
    out = qa_engine.answer([question])
    return render_template('greeting.html', question=question, answer=out[0], 
        detail_ans=json.dumps(out, indent=2))

 
def jsonstr_return(data):
    json_response = json.dumps(data, ensure_ascii = False)
    #creating a Response object to set the content type and the encoding
    response = Response(json_response, content_type="application/json; charset=utf-8" )
    return response 

@app.route('/answer', methods=['POST'])
def translate():
    inputs = request.get_json(force=True)
    question = inputs.get("question")
    logger.info(question)

    out = qa_engine.answer([question])

    return jsonstr_return(out)

@app.route('/answer-debug-detail', methods=['GET', 'POST'])
def translate_debug_detail():
    question = request.args.get("question") 
    logger.info(question)

    out = qa_engine.answer([question])
    
    return jsonstr_return(out)

@app.route('/answer-debug', methods=['GET', 'POST'])
def translate_debug():
    question = request.args.get("question") 
    logger.info(question)

    out = qa_engine.answer([question])
    
    return jsonstr_return(out[0])

def __init_model():
    global qa_engine 
    # qa_engine = WikiQA(model_name_or_path='vasudevgupta/bigbird-roberta-natural-questions')
    # qa_engine = WikiQA('finetuned_models/squad-2.0/robertabase', using_kw_extract=True)
    qa_engine = WikiQALucene(use_msmarco=True, model_name_or_path='finetuned_models/squad-2.0/robertabase_bak')
    # qa_engine = WikiQALucene(use_wiki=True, use_msmarco=False, model_name_or_path='finetuned_models/squad-2.0/robertabase_bak')
     


"""

curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"question": "Who is the first president of US?"}' \
  http://150.65.183.93:7009/answer 


curl --header "Content-Type: application/json" \
--request POST \
--data '{"question": "what the boiling point of water?"}' \
http://150.65.183.93:7009/answer 

"""

if __name__ == "__main__":
    logger.info(__init_model())
    app.run(host=HOST, port=PORT)