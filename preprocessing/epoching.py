import mne


class EEGEpoching:

    @staticmethod
    def create_motor_imagery_epochs(raw):

        events, event_dict = mne.events_from_annotations(raw)

        motor_imagery_events = {
            "left": event_dict["769"],
            "right": event_dict["770"],
            "feet": event_dict["771"],
            "tongue": event_dict["772"],
        }

        epochs = mne.Epochs(
            raw,
            events,
            event_id=motor_imagery_events,
            tmin=0.0,
            tmax=4.0,
            baseline=None,
            preload=True,
            reject_by_annotation=True
        )

        return epochs