import inspect
import logging
import tkinter

import cv2

from utils.util import SteppedIntVar, resize


class CVStep:
    UPDATE_AFTER_CANCEL = True

    def __init__(self, handler, *directargs, show_preview=True):
        self.handler = handler
        self.directargs = directargs
        self.show_preview = show_preview
        self.stepn = 0
        self.valuedict = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__doaftercancel = None
        self.__cb = (None, (), {})

    def set_update_callback(self, handler, *args, **kwargs):
        self.__cb = (handler, args, kwargs)

    def _on_update_trackbar(self):
        self.__doaftercancel = None
        if self.__cb[0]:
            self.__cb[0].__call__(*self.__cb[1], **self.__cb[2])

    def apply_values(self):
        # Make a copy
        vd = dict(self.valuedict)
        for k, v in self.valuedict.items():
            vd[k] = v.get()
        self.logger.debug('tuned param for "%s": %s', self.handler.__name__, str(vd))
        ret = self.handler.__call__(*self.directargs, **vd)

        if self.show_preview:
            img = ret[0] if type(ret) is tuple else ret
            cv2.imshow('[{}]{}'.format(self.stepn, self.handler.__name__), img)

        return ret

    def create_tune_trackbar(self, tk, **tunes):
        # Looking for init value from the handler function
        sig = inspect.signature(self.handler)
        # This is how we define rules for a valid tuning parameter
        tuningparams = list(filter(lambda param: param.default is not param.empty and param.name.startswith('_'),
                                   sig.parameters.values()))
        # Find initial value from param.default
        tuninginitvalues = {}
        for param in tuningparams:
            # print('Parameter:', param, param.default)
            if tunes.get(param.name):
                tuninginitvalues[param.name] = param.default
            else:
                self.logger.warning('no tuning found for "%s". (ignore if it is not a tuning '
                                    'parameter you concerning)', param.name)

        # Delete not used tunes
        for k in list(tunes.keys()):  # make a dict copy
            if k not in [param.name for param in tuningparams]:
                self.logger.warning('tuning param "%s" not found in tuning target `%s`, ignored.', k,
                                    self.handler.__name__)
                del tunes[k]

        # Create trackbars
        for paramname, paramsetting in tunes.items():
            _start = paramsetting[0]
            _end = paramsetting[1]
            _step = paramsetting[2] if len(paramsetting) == 3 else 1
            initval = tuninginitvalues[paramname]
            if _step != 1:
                # use SteppedIntVar
                assert _step > 0
                dynvar = SteppedIntVar(starting=_start, step=_step, value=initval)
            else:
                dynvar = tkinter.DoubleVar(value=initval)
            self.valuedict[paramname] = dynvar

            def _trackbar_callback(_):
                # print('trackbar value changed in step ', self.stepn)
                if self.UPDATE_AFTER_CANCEL:
                    if self.__doaftercancel:
                        tk.after_cancel(self.__doaftercancel)
                    self.__doaftercancel = tk.after(500, self._on_update_trackbar)
                else:
                    self._on_update_trackbar()

            trackbar = tkinter.Scale(tk,
                                     from_=_start, to=_end, variable=dynvar, label=paramname,
                                     command=_trackbar_callback, orient=tkinter.HORIZONTAL, length=500)
            trackbar.pack()
            self.logger.debug('trackbar for "%s" created', paramname)


class CVPipeline:
    """A abstract class that help tuning your OpenCV app pipeline with convenience (wenoptk)"""

    def __init__(self):
        self.steps = []
        self.pipelinename = self.__class__.__name__
        self.logger = logging.getLogger(self.__class__.__name__)

        self._retval = None
        self._current_input = ()

        self._currentstep = 0
        self._tk = tkinter.Tk()
        self._should_create_tuneui = False
        self._flag_update_only = False
        self.__doaftercancel = None

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

    def run_pipeline_tuning(self, *inputargs):
        self._current_input = inputargs
        self._should_create_tuneui = True
        self._flag_update_only = False
        self._currentstep = 0
        self._retval = self._pipeline(*inputargs)
        self._tk.mainloop()
        return self._retval

    def _run_pipeline_update(self):
        """re-run pipeline, only for updating values(tuning params)"""
        assert len(self.steps) > 0
        # Skip blink when initializing sliders
        self.__doaftercancel = None
        self._currentstep = 0
        self._should_create_tuneui = False
        self._flag_update_only = True
        self._retval = self._pipeline(*self._current_input)

    # def run_pipeline_final(self, *inputargs):
    #     self.should_create_tuneui = False
    #     self.isTuning = False
    #     self._currentstep = 0
    #     return self._pipeline(*inputargs)

    def save_tuning(self):
        """Save tuning setting to file"""
        # todo
        pass

    def load_tuning(self):
        # todo
        pass

    def _add_tune_step(self, handler, *directargs, show_preview=True, **kwargs):
        if self._flag_update_only:
            # Load step from memory
            try:
                step = self.steps[self._currentstep]
            except IndexError:
                raise Exception('this step is not loaded yet. Did you call _run_pipeline_update before load steps?')

        else:
            step = CVStep(handler, *directargs, show_preview=show_preview)
            step.stepn = self._currentstep

            def _cb():
                # Skip blink when initializing sliders
                if self.__doaftercancel:
                    self._tk.after_cancel(self.__doaftercancel)
                self.__doaftercancel = self._tk.after(500, self._run_pipeline_update)

            step.set_update_callback(_cb)
            if self._should_create_tuneui:
                step.create_tune_trackbar(self._tk, **kwargs)

            self.logger.debug('step "%s"(n=%d) created', handler.__name__, self._currentstep)

            # Save step info
            self.steps.append(step)

        ret = step.apply_values()

        self._currentstep += 1
        return ret


if __name__ == '__main__':
    """This is an example (as well)"""


    class TuneSomething(CVPipeline):
        def _pipeline(self, *inputargs):
            img_ = inputargs[0]

            def procedure1(img, _valA=5, _valB=12):
                ret1 = {}
                ret2 = _valA + _valB
                print('procedure1: _valA=',_valA,'_valB=',_valB,'ret2==', ret2)
                return img, ret1, ret2

            step1ret = self._add_tune_step(procedure1, img_,
                                           _valA=(0, 100, 1),
                                           _valB=(0, 100, 2),
                                           )

            def procedure2(img, param1, param2, _tune1=12):
                print('procedure2, param1=', param1, 'param2=', param2)
                return img

            step2ret = self._add_tune_step(procedure2, *step1ret, _tune1=(1, 20))

            def p3(img, param1):
                print('procedure3: param1=', param1)
                return img

            step3ret = self._add_tune_step(p3, *(step2ret, 'myparam1'))

            return step3ret

    logging.basicConfig(level=logging.DEBUG, format='%(name)-12s %(levelname)-8s %(message)s')
    t = TuneSomething()
    t.run_pipeline_tuning(
        resize(cv2.imread('../test_dataset/DSC01332.jpg'), 500)
    )
