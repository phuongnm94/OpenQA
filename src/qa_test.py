from wiki_qa import WikiQA
from wiki_qa_lucene import WikiQALucene

if __name__ == "__main__":
    """
    Human: what color of the blood? (100%)
    Robot: " red"
    Human: what color of the mountain? (100%)
    Robot: ""
    Human: what color of the snow? (100%)
    Robot: ""
    Human: what color of human? (100%)
    Robot: ""
    Human: how many people in the world? (100%)
    Robot: ""
    Human: what is the population of the world (100%)
    Robot: " 7.02, 7.06, and 7.08 billion"
    Human: how tall of everest mountain? (100%)
    Robot: " 8,000-meter"
    """
    questions = [
        # 'When was Barack Obama born?',
        # 'Why is the sky blue?',
        # 'How many sides does a pentagon have?',
        # 'what color of the sky?',
        # 'what color of the blood?',
        'Who is the first president of US?',
        # 'When did the COVID-19 pandemic start?',
        'what the boiling point of water?',
        # 'who is Vladimir Lenin?',
        # 'who is albert einstein?',
        # 'when was albert einstein born?',
        'who is Newton?',
        'when was Newton born?',
    ]


    wikiqa = WikiQALucene()
    wikiqa.answer(questions[:1])