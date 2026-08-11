import mne


class EEGEpoching:

    @staticmethod
    def create_motor_imagery_epochs(raw):

        # Extract events from GDF annotations
        events, event_dict = mne.events_from_annotations(raw)

        # Motor imagery event mapping
        motor_imagery_events = {
            "left": event_dict["769"],
            "right": event_dict["770"],
            "feet": event_dict["771"],
            "tongue": event_dict["772"],
        }

        # Create epochs
        #
        # -1 to 0 seconds  -> baseline
        #  0 to 4 seconds  -> motor imagery
        #
        epochs = mne.Epochs(
            raw,
            events,
            event_id=motor_imagery_events,
            tmin=-1.0,
            tmax=4.0,
            baseline=None,
            preload=True,
            reject_by_annotation=True
        )

        return epochs