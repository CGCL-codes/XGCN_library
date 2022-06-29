from utils.utils import combine_dict_list_and_calc_mean

from tqdm import tqdm


def eval_model(model, eval_dl, desc="eval"):
    
    assert hasattr(model, 'eval_a_batch')
    assert hasattr(eval_dl, 'num_samples')
    
    batch_results_list = []
    batch_results_weights = []
    num_samples = eval_dl.num_samples()
    
    for batch_data in tqdm(eval_dl, desc=desc):
        batch_results, num_batch_samples = model.eval_a_batch(batch_data)
        batch_results_list.append(batch_results)
        batch_results_weights.append(num_batch_samples / num_samples)
    
    results = combine_dict_list_and_calc_mean(batch_results_list, batch_results_weights)
    return results
