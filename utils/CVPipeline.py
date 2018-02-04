import configparser
import inspect
import logging
import tkinter
import os

import cv2

from utils.util import SteppedIntVar, resize


class CVStep:
    UPDATE_AFTER_CANCEL = True

    def __init__(self, handler, *directargs, show_preview=True):
        self.valuedict = {}
        self.paramsettingdict = {}
        self.handler = handler
        self.directargs = directargs
        self.show_preview = show_preview
        self.stepn = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.suppress_preview = False

        self.__doaftercancel = None
        self.__cb = (None, (), {})

    def set_update_callback(self, handler, *args, **kwargs):
        self.__cb = (handler, args, kwargs)

    def _on_update_trackbar(self):
        self.__doaftercancel = None
        vd = self.get_actual_valuedict()
        self.logger.debug('"%s" tune params changed: %s', self.handler.__name__, str(vd))
        if self.__cb[0]:
            self.__cb[0].__call__(*self.__cb[1], **self.__cb[2])

    def get_actual_valuedict(self):
        # Make a copy
        vd = dict(self.valuedict)
        for k, v in self.valuedict.items():
            vd[k] = v.get()
        return vd

    def apply_values(self):
        vd = self.get_actual_valuedict()
        self.logger.debug('tuned param for "%s": %s', self.handler.__name__, str(vd))
        ret = self.handler.__call__(*self.directargs, **vd)

        if self.show_preview and not self.suppress_preview:
            img = ret[0] if type(ret) is tuple else ret
            cv2.imshow('[{}]{}'.format(self.stepn, self.handler.__name__), img)

        return ret

    def init_tune_params(self, initoverride: dict=None, **tunes):
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

        # Make override
        if initoverride:
            for k, v in initoverride.items():
                if k not in tuninginitvalues:
                    # Only 'override' existing
                    continue
                self.logger.debug('initial value of "%s" override to %s', k, v)
                tuninginitvalues[k] = v

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
            self.paramsettingdict[paramname] = {
                'from': _start,
                'to': _end,
                'step': _step,
                'dynvar': dynvar
            }

    def create_tune_trackbar(self, tk):
        """Please make sure you have called init_tune_params() before this"""

        for paramname,v in self.paramsettingdict.items():

            def _trackbar_callback(_):
                # print('trackbar value changed in step ', self.stepn)
                if self.UPDATE_AFTER_CANCEL:
                    if self.__doaftercancel:
                        tk.after_cancel(self.__doaftercancel)
                    self.__doaftercancel = tk.after(500, self._on_update_trackbar)
                else:
                    self._on_update_trackbar()

            trackbar = tkinter.Scale(tk,
                                     from_=v['from'], to=v['to'], variable=v['dynvar'], label=paramname,
                                     command=_trackbar_callback, orient=tkinter.HORIZONTAL, length=500)
            trackbar.pack()
            self.logger.debug('trackbar for "%s" created', paramname)


class CVPipeline:
    """A abstract class that help tuning your OpenCV app pipeline with convenience (wenoptk)"""

    def __init__(self):
        self.steps = []
        self.pipelinename = self.__class__.__name__
        self.logger = logging.getLogger(self.__class__.__name__)
        self.configpath = './'
        self.savedconfig = self.load_tuning()

        self._retval = None
        self._current_input = ()

        self._currentstep = 0
        self._tk = tkinter.Tk()
        self._should_create_tuneui = False
        self._flag_update_only = False
        self.__doaftercancel = None
        self._suppress_preview = False

    @property
    def config_url(self):
        return os.path.join(self.configpath, self.pipelinename+'.conf')

    def _pipeline(self, *inputargs):
        """
        Implement your own pipeline here.

            Function for a step can be like this:

            def procedure1(img, foo, bar, _valA=5, _valB=12):
                return img, foo+1

            Parameter starts with a '_' and have a init val will be regarded as a tuning parameter.
            If you want to preview tuned img, you should return the image in the 1st position.

        :param inputargs:
        :return:
        """
        raise NotImplementedError('You should implement your own pipeline here')

    def load_pipeline_quite(self, *inputargs):
        self._flag_update_only = False
        self._suppress_preview = True
        self._should_create_tuneui = False
        self._currentstep = 0
        if len(inputargs) == 0:
            import numpy as np
            h, w = 1, 1
            blank_image = np.zeros((h, w, 3), np.uint8)
            inputargs = (blank_image)
        return self._pipeline(*inputargs)

    def run_pipeline_tuning(self, *inputargs):
        """Run pipeline tuning. Will try to read the params from the config file"""
        self._current_input = inputargs
        self._should_create_tuneui = True
        self._flag_update_only = False
        self._suppress_preview = False
        self._currentstep = 0
        self._create_common_gui()

        self._retval = self._pipeline(*inputargs)
        # Block until tune window closed
        self._tk.mainloop()
        return self._retval

    def run_pipeline_final(self, *inputargs):
        """Run the tuned, final pipeline. Will try to read the params from the config file"""
        self._should_create_tuneui = False
        self._flag_update_only = True
        self._currentstep = 0
        self._suppress_preview = True
        return self._pipeline(*inputargs)

    def _run_pipeline_update(self):
        """re-run pipeline, only for updating values(tuning params)"""
        assert len(self.steps) > 0
        # Skip blink when initializing sliders
        self.__doaftercancel = None
        self._currentstep = 0
        self._should_create_tuneui = False
        self._flag_update_only = True
        self._retval = self._pipeline(*self._current_input)

    def _create_common_gui(self):
        self._tk.title(self.pipelinename)

        # Create a SAVE button
        b = tkinter.Button(self._tk, text="SAVE CURRENT", command=self.save_tuning)
        b.pack()

        def load_default():
            pass

        b = tkinter.Button(self._tk, text="LOAD DEFAULT", command=load_default)
        b.pack()

    def save_tuning(self):
        """Save tuning setting to file"""
        fn = self.config_url
        config = configparser.ConfigParser()
        for step in self.steps:
            sect = step.handler.__name__
            config.add_section(sect)
            vd = step.get_actual_valuedict()
            for k,v in vd.items():
                print('setting config:', step.handler.__name__, k, str(v))
                config.set(sect, k, str(v))
        config.write(open(fn, 'w'))
        self.logger.info('tuning config file saved to "%s"', fn)

    def load_tuning(self) -> configparser.ConfigParser:
        fn = self.config_url
        config = configparser.ConfigParser()
        if os.path.isfile(fn):
            return config
        config.read(fn)
        self.logger.info('loading tuning config file from "%s"', fn)
        # for step in self.steps:
        #     for k,v in step.valuedict.items():
        #         value_ = config.get(step.handler.__name__, k)
        #         step.valuedict[k].set(value_)
        return config

    def _add_tune_step(self, handler, *directargs, show_preview=True, **kwargs):
        if self._flag_update_only:
            # Load step from memory
            try:
                step = self.steps[self._currentstep]
            except IndexError:
                raise Exception('This step is not loaded yet. Did you call _run_pipeline_update before load steps?')

        else:
            step = CVStep(handler, *directargs, show_preview=show_preview)
            step.stepn = self._currentstep

            def _cb():
                # Skip blink when initializing sliders
                if self.__doaftercancel:
                    self._tk.after_cancel(self.__doaftercancel)
                self.__doaftercancel = self._tk.after(500, self._run_pipeline_update)

            step.set_update_callback(_cb)

            # Init tuning parameters
            # Try to read init value from config
            initoverride = {}
            sect = handler.__name__
            if self.savedconfig.has_section(sect):
                for k, v in kwargs.items():
                    if self.savedconfig.has_option(sect, k):
                        initoverride[k] = self.savedconfig.get(sect, k)
            step.init_tune_params(**kwargs, initoverride=initoverride)

            if self._should_create_tuneui:
                step.create_tune_trackbar(self._tk)

            self.logger.debug('step "%s"(n=%d) created', handler.__name__, self._currentstep)

            # Save step info
            self.steps.append(step)

        step.suppress_preview = self._suppress_preview
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
    test_img = resize(cv2.imread('../test_dataset/DSC01332.jpg'), 500)

    # t.run_pipeline_tuning( test_img )

    t.load_pipeline_quite()
    t.run_pipeline_final( test_img )
