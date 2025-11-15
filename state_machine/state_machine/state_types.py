import enum

class StateType(enum.Enum):
    GB_TRACK = 'GB_TRACK' 
    TRAILING = 'TRAILING' 
    OVERTAKE = 'OVERTAKE' 
    FTGONLY = 'FTGONLY'

    TRAILING_TO_GBTRACK = 'TRAILING_TO_GBTRACK'
    # LOW_BAT = 'LOW_BAT'
