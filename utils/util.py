import inspect
import logging
import tkinter

import cv2

logger = logging.getLogger(__name__)


def resize(img, dst_width):
    l = len(img.shape)
    if l == 3:
        h, w, _ = img.shape
    elif l == 2:
        h, w = img.shape
    else:
        raise ValueError
    factor = dst_width / w
    return cv2.resize(img, (0, 0), fx=factor, fy=factor)


class SteppedIntVar(tkinter.IntVar):
    """A stepped characteristic interval variable (a hack for tkinter.IntVar)"""

    def __init__(self, starting=0, step=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.starting = starting
        self.step = step

    def _transform_value(self, value):
        # Offset the init value
        value -= self.starting

        exceed = value % self.step
        _should_move_front = exceed < self.step / 2
        # Move to nearest front value first
        value -= exceed
        if not _should_move_front:
            # Move a whole step backwards
            value += self.step

        # Offset the init value back
        value += self.starting
        return value

    def set(self, value):
        value = self._transform_value(value)
        super().set(value)

    def get(self):
        val = super().get()
        return self._transform_value(val)


class Tuning():
    """
    为opencv的参数调整流程，自动创建trackbar

    by Wenoptk
    """
    UPDATE_AFTER_CANCEL = True

    def __init__(self, callback, img, windowname='myTuningWindow', **kwargs):
        """
        例如要调试的函数为
            myCallback(img, param_1=initVal_1, param_2=initVal_2, param_3=initVal_3, ... )
            注意：调试参数(比如上述的param_n必须要有默认值)

        则应该这样调用
            myTuneWindow(myCallback, img,
                param_1=(初值, 末值[, step]),
                param_2=(初值, 末值[, step])，
                param_3=(初值, 末值[, step]),
                ...
            )
        """
        self.img = img
        self.windowname = windowname
        self.returnval = None
        self.doaftercancel = None
        self.value_dict = {}
        self.target = callback
        self.tk = tkinter.Tk(windowname)
        self.kwargs = kwargs

        # Looking for init value from the callback function
        sig = inspect.signature(callback)
        effectparams = list(filter(lambda param: param.default is not param.empty, sig.parameters.values()))

        for k in list(kwargs.keys()):  # make a copy
            if k not in [param.name for param in effectparams]:
                logger.warning('parameter "%s" not found in tuning target `%s`, ignored.', k, self.target.__name__)
                del kwargs[k]

        # This is where the actual parameter values will be stored
        self.value_dict = dict.fromkeys(kwargs, 0)

        for param in effectparams:
            # print('Parameter:', param, param.default)
            if kwargs.get(param.name):
                self.value_dict[param.name] = param.default
            else:
                logger.warning('no parameter found for "%s" in tuning setting. (ignore if it is not a tuning '
                               'parameter you concerning)', param.name)

        # print(kwargs)
        for paramname, paramtune in kwargs.items():
            _start = paramtune[0]
            _end = paramtune[1]
            _step = paramtune[2] if len(paramtune) == 3 else 1
            initval = self.value_dict[paramname]
            if _step != 1:
                # use SteppedIntVar
                assert _step > 0
                dynvar = SteppedIntVar(starting=_start, step=_step, value=initval)
            else:
                dynvar = tkinter.DoubleVar(self.tk, initval)
            self.value_dict[paramname] = dynvar
            w = tkinter.Scale(self.tk, from_=_start, to=_end, variable=dynvar,
                              label=paramname, command=self._trackbar_callback, orient=tkinter.HORIZONTAL,
                              length=500)
            w.pack()
        pass


    def apply_values(self):
        self.doaftercancel = None
        # Make a copy
        vd = dict(self.value_dict)
        for k, v in self.value_dict.items():
            vd[k] = v.get()
        logger.debug('in "%s" param changed: %s', self.windowname, str(vd))
        self.returnval = self.target.__call__(self.img, **vd)

    def _trackbar_callback(self, _):
        if self.UPDATE_AFTER_CANCEL:
            if self.doaftercancel:
                self.tk.after_cancel(self.doaftercancel)
            self.doaftercancel = self.tk.after(500, self.apply_values)
        else:
            self.apply_values()

    def run(self):
        # Make a call at the run
        self.apply_values()

        # Wait for the tune window being closed
        tkinter.mainloop()
        return self.returnval
