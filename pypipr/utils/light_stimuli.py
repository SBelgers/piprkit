from typing import Optional, Any
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
class LightStimulus:
    """Represents a light stimulus with a start time, end time, and color.
    """    
    def __init__(
        self,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        end_time: Optional[float] = None,
        colour: Optional[str] = None,
    ):
        """Initialize a LightStimulus object.

        Args:
            start_time (float, optional): The start time of the stimulus. Defaults to 0.0.
            duration (Optional[float], optional): The duration of the stimulus. Defaults to None.
            end_time (Optional[float], optional): The end time of the stimulus. Defaults to None.
            colour (Optional[str], optional): The color of the stimulus. Defaults to None.
        """        
        self._set_time(start_time=start_time, duration=duration, end_time=end_time)
        self._set_color(color=colour)
    
    def __repr__(self):
        return f"LightStimulus(start_time={self.start_time}, end_time={self.end_time}, color={self.color})"
        
    def _set_time(self, start_time: float, duration: Optional[float] = None, end_time: Optional[float] = None) -> None:
        if end_time is None and duration is None:
            raise ValueError("Either duration or end_time must be specified.")
        if end_time is not None and duration is not None:
            raise ValueError("Only one of duration or end_time can be specified.")
        self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time
            return
        elif duration is not None:
            self.end_time = start_time + duration
            return
        raise ValueError("Invalid parameters for setting time.")
    
    def _set_color(self, color: Optional[str]) -> None:
        if color is not None:
            from matplotlib.colors import is_color_like
            if not is_color_like(color):
                raise ValueError(f"Invalid colour: {color}")
            self.color = color
        else:
            self.color = "yellow"
        
    # def _set_intensity(self, intensity: Optional[float]) -> None:
    #     if intensity is None:
    #         self.intensity = 1
    #         return
    #     if intensity < 0 or intensity > 1:
    #         raise ValueError("Intensity must be between 0 and 1.")
    #     self.intensity = intensity
    
    def get_time(self) -> tuple[float, float]:
        return self.start_time, self.end_time
    def get_color(self) -> str:
        return self.color

    def set_time_offset(self, offset: float) -> None:
        """Set a time offset for the stimulus. This shifts the start and end times by the specified offset.

        Args:
            offset (float): The offset to apply to the start and end times.
        """        
        self.start_time += offset
        self.end_time += offset

    def plot(
        self,
        ax: Optional[Axes] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot the light stimulus on the given axes.

        Args:
            ax (Optional[Axes], optional): The axes to plot on. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.

        Returns:
            Axes: _description_
        """        
        if ax is None:
            _, ax = plt.subplots()  # type: ignore
        start_time, end_time = self.get_time()

        ax.axvspan(start_time, end_time, alpha=0.3, color=self.get_color(), label='Light Stimulus')  # type: ignore
        ax.axvline(start_time, color='black', linestyle='--', alpha=0.7, label='_Light Onset')  # type: ignore
        ax.axvline(end_time, color='black', linestyle='--', alpha=0.7, label='_Light Offset')  # type: ignore
        if show:
            plt.show()  # type: ignore
        return ax


class LightStimuliSeries:
    """Represents a series of light stimuli.
    """    
    def __init__(self):
        """Initialize a LightStimuliSeries object.
        """        
        self.stimuli: list[LightStimulus] = []
        
    def _add_stimulus_from_class(self, stimulus: LightStimulus) -> None:
        self.stimuli.append(stimulus)
        
    def add_stimulus(self, start_time: float, duration: Optional[float] = None, end_time: Optional[float] = None, colour: Optional[str] = None) -> None:
        """Add a light stimulus to the series.

        Args:
            start_time (float): The start time of the light stimulus.
            duration (Optional[float], optional): The duration of the light stimulus. Defaults to None.
            end_time (Optional[float], optional): The end time of the light stimulus. Defaults to None.
            colour (Optional[str], optional): The color of the light stimulus. Defaults to None.
        """        
        stimulus = LightStimulus(start_time=start_time, duration=duration, end_time=end_time, colour=colour)
        self._add_stimulus_from_class(stimulus)

    def get_stimuli(self) -> list[LightStimulus]:
        """Get the list of light stimuli in the series. """
        return self.stimuli
    
    def get_times(self) -> list[tuple[float, float]]:
        """Get the start and end times of all light stimuli in the series. """
        return [stimulus.get_time() for stimulus in self.stimuli]
    def get_colors(self) -> list[str]:
        """Get the colors of all light stimuli in the series. """
        return [stimulus.get_color() for stimulus in self.stimuli]

    def set_time_offset(self, offset: float) -> None:
        """Set a time offset for all light stimuli in the series. This shifts the start and end times of each stimulus by the specified offset."""
        for stimulus in self.stimuli:
            stimulus.set_time_offset(offset)
    def plot(
        self,
        ax: Optional[Axes] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot all light stimuli in the series on the given axes. """
        if ax is None:
            _, ax = plt.subplots()  # type: ignore
        for stimulus in self.stimuli:
            stimulus.plot(ax=ax, show=False, **kwargs)
        if show:
            plt.show() # type: ignore
        return ax