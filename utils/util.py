import inspect

import cv2


def myTuneWindow(callback, img, **kwargs):
    '''
    为opencv的参数调整流程，自动创建trackbar

    例如要调试的函数为 myThreshold(img, param_1=initVal_1, param_2=initVal_2, param_3=initVal_3 )
    则应该这样调用
        myTuneWindow(myThreshold, img, param_1=(初值, 末值[, step]), param_2=(初值, 末值[, step])， param_2=(初值, 末值[, step]))

    by Wenop

    :param callback:
    :param img:
    :param kwargs:
    :return:
    '''

    def _cb(x):
        for k, v in kwargs.items():
            _start = v[0]
            _end = v[1]
            _step = v[2] if len(v) == 3 else 1
            # 获取trackbar位置并把值映射回去
            valueDict[k] = cv2.getTrackbarPos(k, 'tune') * _step + _start

        print('param changed:', valueDict)

        callback(img, **valueDict)

    cv2.namedWindow('tune', cv2.WINDOW_NORMAL)

    # This is where the actual parameter values will be stored
    valueDict = dict.fromkeys(kwargs, 0)

    # Looking for init value from the callback function
    sig = inspect.signature(callback)
    for param in sig.parameters.values():
        if param.default is not param.empty:
            # print('Parameter:', param, param.default)
            if kwargs.get(param.name):
                valueDict[param.name] = param.default

    # print(kwargs)
    for k, v in kwargs.items():
        _start = v[0]
        _end = v[1]
        _step = v[2] if len(v) == 3 else 1
        cv2.createTrackbar(k, 'tune', int((valueDict[k] - _start) / _step), int((_end - _start) / _step), _cb)

    cv2.resizeWindow('tune', 600, 9 * len(kwargs))

    # Make a call at the run
    callback(img, **valueDict)

    # Wait for the tune window being closed
    while cv2.getWindowProperty('tune', 0) >= 0:

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
