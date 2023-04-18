.. _user_guide-training_and_evaluation-model_inference:

Model Inference
==================

XGCN provides some model inference APIs: 

.. code:: python

    # infer scores given a source node and one or more target nodes:
    target_score = model.infer_target_score(
        src=5, 
        target=torch.LongTensor(101, 102, 103)
    )

    # infer top-k recommendations for a source node
    score, topk_node = model.infer_topk(k=100, src=5, mask_nei=True)

    # save the output embeddings as a text file
    model.save_emb_as_txt(filename='out_emb_table.txt')
