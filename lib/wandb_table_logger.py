import wandb

def format_sentences(sentences):
    formatted_output = ""

    for sentence in sentences:
        formatted_output += "-'" + sentence + "'\n"

    return formatted_output

class WandbTableLogger(object):
    def __init__(self, name):
        self.name = name

        self.columns = ["epoch", "scene_id", "object_id", "object_type", "sentence_id", "gt", "pred"]

        self.table = wandb.Table(columns=self.columns,)
        wandb.Table.MAX_ROWS = 1_000_000

        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def add_data(self, corpus, candidates):
        for key in corpus.keys():
            split_key = key.split("|")
            scene_id = split_key[0]
            object_id = split_key[1]
            object_type = split_key[2]

            if len(candidates[key]) > 1:
                raise NotImplementedError()

            candidate_sentence = candidates[key][0]

            for i, corpus_sentence in enumerate(corpus[key]):
                self.table.add_data(self.epoch, scene_id, object_id, object_type, i,corpus_sentence, candidate_sentence)

    def log(self):
        wandb.log({"tables/"+self.name: self.table})   

