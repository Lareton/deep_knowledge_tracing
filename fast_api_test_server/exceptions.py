
class TestItemsAlreadyUploaded(Exception):
    def __init__(self, message="Test items already uploaded into table"):
        self.message = message
        super().__init__(self.message)