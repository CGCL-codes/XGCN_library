from .OnePosKNeg_Evaluator import OnePosKNeg_Evaluator
from .OnePosWholeGraph_Evaluator import OnePosWholeGraph_Evaluator
from .MultiPosWholeGraph_Evaluator import MultiPosWholeGraph_Evaluator


def select_evaluator(evaluation_method):
    return {
        'one_pos_k_neg': OnePosKNeg_Evaluator,
        'one_pos_whole_graph': OnePosWholeGraph_Evaluator,
        'whole_pos_whole_graph': MultiPosWholeGraph_Evaluator,
    }[evaluation_method]


def create_val_Evaluator(config, data, model):
    Evaluator = select_evaluator(config['val_method'])
    evaluator = Evaluator(model, 
                          file_eval_set=config['file_val_set'],
                          batch_size=config['val_batch_size'])
    return evaluator


def create_test_Evaluator(config, data, model):
    Evaluator = select_evaluator(config['test_method'])
    evaluator = Evaluator(model, 
                          file_eval_set=config['file_test_set'],
                          batch_size=config['test_batch_size'])
    return evaluator
