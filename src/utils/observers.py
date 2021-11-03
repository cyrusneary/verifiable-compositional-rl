class ObserverIncrementTaskSuccessCount:
    """
    Observer class that updates a counter each time a message is received 
    stating that the agent was successful in completing the task at hand.

    Parameters
    ----------
    observable : CustomSideChannel object
        An observable side channel which acts as the observable, notifying
        this observer each time a new message is received from the environment.
    """

    def __init__(self, observable):
        observable.subscribe(self)
        self.success_count = 0

    def notify(self, observable, message):
        if message == 'Completed task':
            self.success_count = self.success_count + 1