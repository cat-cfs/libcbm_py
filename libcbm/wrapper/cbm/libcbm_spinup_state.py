

class LibCBM_SpinupState:
    """Wrapper for low level enum of the same name defined in LibCBM C/C++ code
    """
    HistoricalRotation = 0,
    HistoricalDisturbance = 1,
    LastPassDisturbance = 2,
    GrowToFinalAge = 3,
    Delay = 4,
    Done = 5

    @staticmethod
    def get_name(x):
        """gets the name of the enum field associated with the specified
        integer

        Args:
            x (int): an integer matching the value of one of the enum fields

        Raises:
            ValueError: raised when the specified value is not a defined enum
                field

        Returns:
            str: the name of the enum field associated with the specified
                integer
        """
        if x == 0:
            return"HistoricalRotation"
        elif x == 1:
            return "HistoricalDisturbance"
        elif x == 2:
            return "LastPassDisturbance"
        elif x == 3:
            return "GrowToFinalAge"
        elif x == 4:
            return "Delay"
        elif x == 5:
            return "Done"
        else:
            raise ValueError("invalid spinup state code")
