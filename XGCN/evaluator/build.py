from .WholeGraph_MultiPos_Evaluator import WholeGraph_MultiPos_Evaluator
from .WholeGraph_OnePos_Evaluator import WholeGraph_OnePos_Evaluator


def select_evaluator(evaluator_type):
    return {
        'WholeGraph_MultiPos_Evaluator': WholeGraph_MultiPos_Evaluator,
        'WholeGraph_OnePos_Evaluator': WholeGraph_OnePos_Evaluator,
    }[evaluator_type]


def build_val_Evaluator(config, data, model):
    Evaluator = select_evaluator(config['val_evaluator'])
    evaluator = Evaluator(model, 
                          file_eval_set=config['file_val_set'],
                          batch_size=config['val_batch_size'])
    return evaluator


def build_test_Evaluator(config, data, model):
    Evaluator = select_evaluator(config['test_evaluator'])
    evaluator = Evaluator(model, 
                          file_eval_set=config['file_test_set'],
                          batch_size=config['test_batch_size'])
    return evaluator
