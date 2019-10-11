

def event_iterator(time_step, sit_events):

    # TODO: In CBM-CFS3 events are sorted by default disturbance type id
    # (ascending) In libcbm, sort order needs to be explicitly defined in
    # cbm_defaults (or other place)
    time_step_events = sit_events[
        sit_events.time_step == time_step]
    for _, time_step_event in time_step_events.itterows():
        yield dict(time_step_event)

