from utils import io
from dataloader.eval_dataloader import \
    WholeGraph_OnePos_EvalDataLoader, \
    WholeGraph_MultiPos_EvalDataLoader, \
    SampledNeg_EvalDataLoader


def build_val_test_dl(config, data):
    if 'not_eval' in config and config['not_eval']:
        return None, None
    
    all_eval_method = [
        'one_pos_whole_graph',
        'multi_pos_whole_graph',
        'one_pos_k_neg'
    ]
    validation_method = config['validation_method']
    test_method = config['test_method']
    
    assert validation_method in all_eval_method
    assert test_method in all_eval_method
    
    val_batch_size = config['val_batch_size'] if 'val_batch_size' in config else 100
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config else 100
    
    val_dl = build_eval_dl(validation_method, 
                           config['file_validation'], 
                           config['mask_nei_when_validation'], 
                           val_batch_size,
                           data)
    
    test_dl = build_eval_dl(test_method, 
                            config['file_test'],
                            config['mask_nei_when_test'],
                            test_batch_size,
                            data)
    
    return val_dl, test_dl


def build_eval_dl(eval_method, file_eval, mask_nei, batch_size, data):
    eval_dl = {
        'one_pos_whole_graph': build_WholeGraph_OnePos_EvalDataLoader,
        'multi_pos_whole_graph': build_WholeGraph_MultiPos_EvalDataLoader, 
        'one_pos_k_neg': build_SampledNeg_EvalDataLoader
    }[eval_method](file_eval, mask_nei, batch_size, data)
    
    return eval_dl


def build_WholeGraph_OnePos_EvalDataLoader(file_eval, mask_nei, batch_size, data):
    pos_edges = io.load_pickle(file_eval)
    
    dl = WholeGraph_OnePos_EvalDataLoader(
        pos_edges=pos_edges,
        batch_size=batch_size, 
        model=data['model'], 
        mask_neighbor_score=mask_nei
    )
    return dl


def build_WholeGraph_MultiPos_EvalDataLoader(file_eval, mask_nei, batch_size, data):
    eval_set = io.load_pickle(file_eval)
    
    dl = WholeGraph_MultiPos_EvalDataLoader(
        src=eval_set['src'], 
        pos_list=eval_set['pos_list'],
        batch_size=batch_size,
        model=data['model'],
        mask_neighbor_score=mask_nei
    )
    return dl


def build_SampledNeg_EvalDataLoader(file_eval, mask_nei, batch_size, data):
    assert not mask_nei, 'don\'t consider mask neighbors\' scores when using 1-pos-k-neg'
    
    src_pos_neg = io.load_pickle(file_eval)
    
    dl = SampledNeg_EvalDataLoader(
        src_pos_neg=src_pos_neg,
        batch_size=batch_size,
        model=data['model']
    )
    return dl
