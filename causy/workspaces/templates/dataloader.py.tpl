from causy.data_loader import AbstractDataLoader

class DataLoader(AbstractDataLoader):
    """
    A causy dataloader.
    
    """
    
    def __init__(self):
        """
        Initialize your dataloader here
        """
        pass
    
    def load(self):
        """
        Load the data. This function should yield the data row by row. You have to yield it as a dictionary of column names and values (floats).
        :return: 
        """
        yield {
            "column1": 1.0,
            "column2": 2.0
        }
        
    def hash(self):
        """
        Returns a hash of the data. This is useful to check if the data has changed.
        :return: 
        """
        return None
