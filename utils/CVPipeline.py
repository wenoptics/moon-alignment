import inspect
import logging

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class TuneView:
    def __init__(self, fig, ax):
        self.ax = ax
        self.fig = fig

    def img(self, obj):
        self.obj = obj
        self.ax.imshow(...)
        _vars = obj.get_variables()
        plt.subplots_adjust(bottom=0.03*(len(_vars)+2))
        self.sliders = []
        for i,var in enumerate(_vars):
            self.add_slider(i*0.03, var[0], var[1], var[2])
        plt.show()

    def add_slider(self, pos, name, min, max):
        ax = plt.axes([0.1, 0.02+pos, 0.8, 0.02], axisbg='lightgoldenrodyellow')
        slider = Slider(ax, name, min, max, valinit=getattr(self.obj, name))
        self.sliders.append(slider)

        def update(val):
            setattr(self.obj, name, val)
            self.l[0].set_ydata(self.obj.series())
            self.fig.canvas.draw_idle()
        slider.on_changed(update)


class CVPipeline:
    """A abstract class that help tuning your OpenCV app pipeline with convenience (wenoptk)"""
    def __init__(self):
        self._tuned_params = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.isTuning = False
        self.__gridnx = 3
        self.__previewgrids = ((), ())

    def _pipeline(self, *inputargs):
        """
        Implement your own pipeline here.

            Function for a step can be like this:

            def procedure1(img, foo, bar, _valA=5, _valB=12):
                pass

            Parameter starts with a '_' and have a init val will be regarded as a tuning parameter.

        :param inputargs:
        :return:
        """
        raise NotImplementedError('You should implement your own pipeline here')

    def _init_preview_grid(self, x, y):
        self.__gridnx = x
        self.__gridny = y
        f, self.__previewgrids = plt.subplots(x, y, sharex='col', sharey='row')

    def run_pipeline_tuning(self, *inputargs):
        self._tuned_params.clear()
        self.isTuning = True
        self._currentstep = 0
        ret = self._pipeline(*inputargs)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
        return ret

    def run_pipeline_final(self, *inputargs):
        self.isTuning = False
        self._currentstep = 0
        return self._pipeline(*inputargs)

    def save_tuning(self):
        """Save tuning setting to file"""
        # todo
        pass

    def load_tuning(self):
        # todo
        pass

    def _add_tune_step(self, handler, *directargs, show_preview=True, **kwargs):
        # self.pipeline_steps.append({
        #     'handler': handler,
        #     'directargs': directargs,
        #     'show_preview': show_preview
        # })
        ax = None
        if self.isTuning:
            # Create a preview plot for this step
            ax = self.__previewgrids[self.__getgridy(self._currentstep)][self.__getgridx(self._currentstep)]

            # Looking for init value from the handler function
            sig = inspect.signature(handler)
            # This is how we define rules for a valid tuning parameter
            tunningparams = list(filter(lambda param: param.default is not param.empty and param.name.startswith('_'),
                                        sig.parameters.values()))

            for k in list(kwargs.keys()):  # make a dict copy
                if k not in [param.name for param in tunningparams]:
                    self.logger.warning('parameter "%s" not found in tuning target `%s`, ignored.', k, handler.__name__)
                    del kwargs[k]

            # Create trackbars etc.
            pos = 0
            for paramname, paramsetting in kwargs.items():
                def _cb():
                    pass

                pos += 1
                axslider = plt.axes([0.25, 0.1+pos*0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
                s = Slider(axslider, paramname, 0.1, 30.0, valinit=0)
                s.on_changed(_cb)

            tuning_params = dict.fromkeys(kwargs, 0)
            # Save the current params
            self._tuned_params.append(tuning_params)
        else:
            # Load from saved params
            tuning_params = self._tuned_params[self._currentstep]
        self._currentstep += 1
        ret = handler(*directargs, **tuning_params)
        if self.isTuning and show_preview:
            img = ret[0]
            ax.imshow(img)
        return ret

    def __getgridy(self, val) -> int:
        return (val - self.__getgridx(val)) // self.__gridnx

    def __getgridx(self, val) -> int:
        return val % self.__gridnx


if __name__ == '__main__':
    """This is an example (as well)"""

    class TuneSomething(CVPipeline):
        def _pipeline(self, *inputargs):
            self._init_preview_grid(2, 3)
            img = inputargs[0]

            def procedure1(img, _valA=5, _valB=12):
                ret1 = {}
                ret2 = _valA + _valB
                print('procedure1: ret2==', ret2)
                return img, ret1, ret2

            step1ret = self._add_tune_step(procedure1, img,
                                           _valA=(0, 100, 2),
                                           _valB=(0, 100, 2),
                                           )

            def procedure2(img, param1, param2, _tune1=12):
                print('procedure2, param1=',param1,'param2=',param2)
                return img

            step2ret = self._add_tune_step(procedure2, *step1ret, _tune1=(1, 20))

            def p3(img, param1):
                print('procedure3: param1=', param1)
                return img
            step3ret = self._add_tune_step(p3, *(step2ret, 'myparam1'))

            return step3ret

    t = TuneSomething()
    t.run_pipeline_tuning(cv2.imread('../test_dataset/DSC01332.jpg'))