class BaseModel:
    
    def __init__(self, config, data):
        pass
    
    def forward_and_backward(self, batch_data):
        # Given a batch of training data,
        # perform forward process to calculate loss, 
        # and then perform backward process to update model parameters.
        
        # Return loss value.
        loss = 0.0
        return loss
    
    def eval(self, batch_data):
        # This function will be called by the Evaluator.
        # The return value should correspond to the Evaluator.
        output = None
        return output
    
    def save(self, root=None):
        # Save the whole model for future inference.
        pass
    
    def load(self, root=None):
        # Load the whole model (to conduct inference).
        pass
