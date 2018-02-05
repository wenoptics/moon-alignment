import configparser
import inspect
import logging
import tkinter
import os

import cv2

from utils.util import SteppedIntVar, resize, isCvWindowsExists


class CVStep:
    UPDATE_AFTER_CANCEL = True

    def __init__(self, from_pipeline: 'CVPipeline', handler, *directargs, show_preview=True):
        self.valuedict = {}
        self.paramsettingdict = {}
        self.handler = handler
        self.directargs = directargs
        self.show_preview = show_preview
        self.stepn = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.frompipeline = from_pipeline

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
        vd = dict(self.valuedict)  # Make a copy
        for k, v in self.valuedict.items():
            vd[k] = v.get()
        return vd

    def apply_values(self):
        """Apply tuning value to the target handler. And show the preview if need to"""
        vd = self.get_actual_valuedict()
        self.logger.debug('tuned param for "%s": %s', self.handler.__name__, str(vd))
        ret = self.handler.__call__(*self.directargs, **vd)

        if self.show_preview and not self.frompipeline._suppress_preview:
            img = ret[0] if type(ret) is tuple else ret
            _winname = '[{}]{}'.format(self.stepn, self.handler.__name__)

            # Resize preview image
            if self.frompipeline.force_resize_preview_w > 0:
                img = resize(img, self.frompipeline.force_resize_preview_w)
                _winname += '(resized)'
            is_existed = isCvWindowsExists(_winname)
            cv2.imshow(_winname, img)

            def get_win_position(index) -> (int, int):
                PADDING_LEFT, PADDING_TOP = 10, 10
                GAP_LEFT, GAP_TOP = 0, 20
                imgw, imgh = img.shape[1], img.shape[0]
                scrw, scrh = self.frompipeline._tk.winfo_screenwidth(), self.frompipeline._tk.winfo_screenheight()
                NX = (scrw - PADDING_LEFT) // (imgw + GAP_LEFT)
                ix = index % NX
                iy = index // NX
                x = ix * (imgw + GAP_LEFT) + PADDING_LEFT
                y = iy * (imgh + GAP_TOP) + PADDING_TOP
                return x, y

            # Arrange the window... when open many windows, it gets so messy
            if not is_existed:
                # Don't arrange the window if user have move it..
                cv2.moveWindow(_winname, *get_win_position(self.stepn))

        return ret

    def init_tune_params(self, initoverride: dict = None, **tunes):
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
                self.logger.warning('no tuning found for "%s". (ignore if it is not a tuning parameter you concerning)',
                                    param.name)

        # Make override
        if initoverride:
            for k, v in initoverride.items():
                if k not in tuninginitvalues:
                    # Only 'override' existing
                    continue
                self.logger.debug('initial value of "%s" override to %s', k, v)
                tuninginitvalues[k] = v

        # Check tune settings
        for k, v in list(tunes.items()):
            if type(v) is not tuple or len(v) < 2:
                self.logger.warning('param "%s" doesnt seem like valid tuning param setting for target `%s`, ignored.',
                                    k, self.handler.__name__)
                del tunes[k]
            if k not in [param.name for param in tuningparams]:
                raise ValueError('Param tuning setting "{0}" doesnt have a related param in `{1}`. '
                                 'Did you forget to set an init value in `{1}`?'
                                 .format(k, self.handler.__name__))

        # Create trackbars
        for paramname, paramsetting in tunes.items():
            start_ = paramsetting[0]
            end_ = paramsetting[1]
            step_ = paramsetting[2] if len(paramsetting) > 2 else 1
            initval = tuninginitvalues[paramname]
            type_set = {type(start_), type(end_), type(step_), type(initval)}
            type_ = float if float in type_set else int
            if step_ != 1:
                # use SteppedIntVar
                assert step_ > 0
                dynvar = SteppedIntVar(starting=start_, step=step_, value=initval)
                type_ = int
            else:
                dynvar = tkinter.DoubleVar(value=initval)
            self.valuedict[paramname] = dynvar
            self.paramsettingdict[paramname] = {
                'from': start_,
                'to': end_,
                'step': step_,
                'dynvar': dynvar,
                'type': type_
            }

    def create_tune_trackbars(self, tk):
        """Please make sure you have called init_tune_params() before this"""
        for paramname, v in self.paramsettingdict.items():

            def _trackbar_callback(_):
                # print('trackbar value changed in step ', self.stepn)
                if self.UPDATE_AFTER_CANCEL:
                    if self.__doaftercancel:
                        tk.after_cancel(self.__doaftercancel)
                    self.__doaftercancel = tk.after(500, self._on_update_trackbar)
                else:
                    self._on_update_trackbar()

            resolution_ = {}
            if v['type'] is float:
                resolution_ = {'resolution': (v['to'] - v['from']) / 100}
            trackbar = tkinter.Scale(tk,
                                     from_=v['from'], to=v['to'], variable=v['dynvar'], label=paramname,
                                     command=_trackbar_callback, orient=tkinter.HORIZONTAL, length=500, **resolution_)
            trackbar.pack()
            self.logger.debug('trackbar for "%s" created', paramname)


class CVPipeline:
    """A abstract class that help tuning your OpenCV app pipeline with convenience (wenoptk)

        Call .run_pipeline_tuning() to tune your pipeline!
        Call .run_pipeline_final() to run your tuned pipeline!
    """

    def __init__(self, force_resize_preview_w=0):
        """

        :param force_resize_preview_w: Force preview img to dst width. 0 for no scale
        """
        self.force_resize_preview_w = force_resize_preview_w
        self.pipelinename = self.__class__.__name__
        self.configpath = './'
        self.logger = logging.getLogger(self.__class__.__name__)
        self.savedconfig = self.load_tuning()
        self.steps = []

        self._retval = None
        self._current_input = ()

        self._currentstep = 0
        self._timesrun = 0
        self._tk = tkinter.Tk()
        self._should_create_tuneui = False
        self._load_steps_only = False
        self._suppress_preview = False
        self.__doaftercancel = None

    @property
    def config_url(self):
        return os.path.join(self.configpath, self.pipelinename + '.conf')

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

    # def load_pipeline_quite(self, *inputargs):
    #     self._load_steps_only = False
    #     self._suppress_preview = True
    #     self._should_create_tuneui = False
    #     if len(inputargs) == 0:
    #         # Default to pass a 1x1 dummy image
    #         import numpy as np
    #         h, w = 1, 1
    #         blank_image = np.zeros((h, w, 3), np.uint8)
    #         inputargs = (blank_image)
    #     return self.__run(*inputargs)

    def run_pipeline_tuning(self, *inputargs):
        """Run pipeline tuning. Will try to read the params from the config file

            Noted that this will invoke the pipeline twice (one for initializing all steps,
                one invoked when tk.trackbars initialized)
        """
        self._current_input = inputargs
        self._should_create_tuneui = True
        self._load_steps_only = False
        self._suppress_preview = False

        self._create_common_gui()
        # First run to create trackbar ui etc.
        self._retval = self.__run(*inputargs)

        # Block until tune window closed
        self._tk.mainloop()
        return self._retval

    def run_pipeline_final(self, *inputargs):
        """Run the tuned, final pipeline. Will try to read the params from the config file"""
        if self._timesrun == 0:
            # This is the first time to run, load steps first.
            self._load_steps_only = False
        else:
            # Steps are already loaded, we won't create new steps, just load them
            self._load_steps_only = True
        self._suppress_preview = True
        return self.__run(*inputargs)

    def _run_pipeline_update(self):
        """re-run pipeline, only for updating values(tuning params)"""
        assert len(self.steps) > 0
        self._load_steps_only = True
        self._should_create_tuneui = False
        # Skip blink when initializing sliders
        self.__doaftercancel = None
        self._retval = self.__run(*self._current_input)

    def __run(self, *inputargs):
        self._pre_pipeline()
        self._retval = self._pipeline(*inputargs)
        self._post_pipeline()
        return self._retval

    def _pre_pipeline(self):
        if self._load_steps_only:
            if self._should_create_tuneui:
                raise ValueError('_should_create_tuneui will not have effect when _load_steps_only set')
        self._currentstep = 0

    def _post_pipeline(self):
        self._timesrun += 1

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
            for k, v in vd.items():
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
        if self._load_steps_only:
            # Load step from memory
            try:
                step = self.steps[self._currentstep]
            except IndexError:
                raise Exception('This step is not loaded yet. Did you call _run_pipeline_update before load steps?')

        else:
            step = CVStep(self, handler, *directargs, show_preview=show_preview)
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
                labelframe = tkinter.LabelFrame(self._tk, text=handler.__name__)
                labelframe.pack(expand="yes")

                step.create_tune_trackbars(labelframe)

            # Save step info
            self.steps.append(step)
            self.logger.debug('step "%s"(n=%d) created', handler.__name__, self._currentstep)

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
                print('procedure1: _valA=', _valA, '_valB=', _valB, 'ret2==', ret2)
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

    t.run_pipeline_final(test_img)
