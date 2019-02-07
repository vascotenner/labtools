"""
Several tools to interact with lab equipement
First versions in 20150909_measurer, 20150914_measurer.ipynb
(c) Vasco Tenner 2015, GPLv3 licence
"""

from pyqtgraph.Qt import QtGui, QtCore
import queue
from queue import Queue
import time

import numpy as np
import pyqtgraph as pg

try:
    import pyfits
except ImportError:
    import astropy.io.fits as pyfits
import collections

from filename_tools.find_available_filename import find_available_filename, find_available_number
Slot = QtCore.pyqtSlot

Measurement = collections.namedtuple('Measurement', ['header', 'data'])

def create_header(cam1='cam'):
    """Get information about settings of xenics linear array and stage and 
    return as dict"""

    header = create_cam_header(cam1=cam1)

    if 'stage' in globals(): header.update({'position_{}'.format(ax): pos.__str__() for ax,pos in
                                            enumerate(stage._position_cached)})
    if 'piezo' in globals():header.update({'piezo': piezo.position})
    if 'axisz' in globals():header.update({'axisz': axisz.position})
    if 'delay' in globals():header.update({'delay': delay.position})
    header.update({'time':time.time()})
    return header

def create_cam_header(cam1='cam'):
    """Create header with information from camera cam"""
    names = ['exposure_time', 'gain', 'OffsetY', 'OffsetX',]
    cam = globals()[cam1]
    header = {name: cam.__getattribute__(name).__str__() for name in names}
    return header

def create_hdu(frame=None):
    """Convert a Measurement to a pyfits object"""
    if frame is None:
        frame = frames[-1]
    hdu = pyfits.ImageHDU(frame.data)
    [hdu.header.set(name,val.__str__() if hasattr(val, 'units') else val) for name,val in frame.header.items()]
    return hdu

def saver(frame=None, fname=None, verbose=True, zeros=3):
    """Save a list of Measurements to fits"""

    if type(frame) == list:
        hdu = [create_hdu(f) for f in frame]
    else:
        hdu = [create_hdu(frame)]
    if not type(hdu[0]) == pyfits.PrimaryHDU:
        hdu[0] = pyfits.PrimaryHDU(hdu[0].data, hdu[0].header)
    hdu = pyfits.HDUList(hdu)
    if fname is None:
        fname = find_available_filename('_frame', extension='fits',zeros=zeros)
    hdu.writeto(fname)
    if verbose:
        print('Written to', fname)
    return fname

def flatten(d):
    '''Join a dict to k=val_k2=val2_....'''
    return '_'.join(['='.join([k,  '{:.2f}'.format(d[k])
                               if isinstance(d[k], float)  # does not work for
                                                           # quantity
                               else str(d[k])
                              ])
                     for k in d])

def saver_sep(frame=None, fname='', steps=None, verbose=True, zeros=3):
    """Save a list of Measurements to individual fits files"""
    fnames = []
    if not frame:
        frame = frames2
    if not steps:
        for f in frame:
            fn = find_available_filename('_{}'.format(
                fname), extension='fits', zeros=zeros)
            fnames.append(fn)
            saver(f, fn, verbose=verbose, zeros=zeros)
        return fnames
    else:
        frames = list(frame)[-len(steps):]
        for step, frame in zip(steps, frames):
            if fname:
                fn = find_available_filename('_{}_{}'.format(
                    fname, flatten(step)), extension='fits', zeros=zeros)
                fn = '{}_{}_{}.fits'.format(
                    find_available_number(), fname, flatten(step))
            else:
                fn = find_available_filename('_{}'.format(
                    flatten(step)), extension='fits', zeros=zeros)
            fnames.append(fn)
            saver(frame, fn, verbose=verbose, zeros=zeros)
        return fnames

def saver_scan(scan, verbose=True, zeros=3):
    return saver_sep(frame=scan.frames, fname=scan.name, steps=scan.steps,
                     verbose=verbose, zeros=zeros)

@Slot(object)
def collector2(measurement):
    global frames2
    frames2.append(measurement)

def myGetQTableWidgetSize(t):
    w = t.verticalHeader().width() + 4 # +4 seems to be needed
    for i in range(t.columnCount()):
        w += t.columnWidth(i) # seems to include gridline (on my machine)
    h = t.horizontalHeader().height() + 4
    for i in range(t.rowCount()):
        h += t.rowHeight(i)
    return QtCore.QSize(w, h)

class Measurer(QtCore.QObject):
    ''' Represents a punching bag; when you punch it, it
        emits a signal that indicates that it was punched. '''
    punched = QtCore.Signal(object)
    timer = QtCore.QTimer()
    
    def __init__(self, capture, cam='cam'):
        # Initialize the PunchingBag as a QObject
        QtCore.QObject.__init__(self)
        self.cont = True
        self.capture = capture
        self.cam = cam
        self.loop = False
    
    def start(self, loop=True):
        self.cont = True
        self.loop = loop
        self.measure()
        
    def measure(self):
        ''' Punch the bag '''
        #global frames
        y = self.capture()
        m = Measurement(create_header(cam1=self.cam), y)
        #frames.append(m)
        self.punched.emit(m)
        if self.loop and self.cont:
            self.timer.singleShot(1,self.measure)
        return m
    
    def stop(self):
        self.cont = False
            

class ScanCore(QtCore.QObject):
    ''' Core of a scanobject
    
    params:
    prebody: (waittime [ms], function(step) ) '''

    result = QtCore.Signal(object)
    result_and_step = QtCore.Signal(object)
    finished = QtCore.Signal(object)
    timer = QtCore.QTimer()
    
    def __init__(self, body=None, steps=None, iterer=None, prebody=None,
                 onfinished=None, name='scan', estimate=0):
        super().__init__()
        self.name = name
        self.steps = steps
        self.iterer = iterer
        self.paused = False
        self.body = body
        self.prebody = prebody
        self.onfinished = onfinished
        self._ready = False
        self.ready_time = None
        self.start_time = None
        self._estimate = estimate

    def __repr__(self):
        return '{}: estimated duration {}s'.format(self.name, self.estimate)

    def reset(self):
        '''Reset status such that rerun is posible'''
        self.iterer = None
        self.ready = False
        self.start_time = None

    @property
    def estimate(self):
        if self.start_time and not self.ready and self.count > 0:
            return (len(self.steps) - self.count)/self.count * \
                        (0.0001+(time.time() - self.start_time))
        else:
            return self._estimate

    @property
    def ready(self):
        return self._ready

    @ready.setter
    def ready(self, status):
        if status:
            self._ready = True
            self.ready_time = time.time()
            self._estimate = self.ready_time - self.start_time
            self.finished.emit(status)
        else:
            self._ready = status

    def start(self, steps=None):
        if self.ready:
            raise RuntimeError('Scan has run already, reset first')
        self.cont = True
        if not self.paused:
            if not steps is None:
                self.steps = steps
                self.iterer = self.steps.__iter__()
            if self.iterer is None:
                self.iterer = self.steps.__iter__()
        
            self.start_time = time.time()
            self.count = 0
            def internal():
                try:
                    if self.cont:
                        step = self.iterer.__next__()
                        self.count += 1

                        def measureandcontinue():

                                #connect to saver
                                @Slot(object)
                                def add_step_and_emit(measurement):
                                    self.result_and_step.emit((step, measurement))
                                self.result.connect(add_step_and_emit)

                                self.body(self.result, step)
                                self.result.disconnect(add_step_and_emit)
                                self.timer.singleShot(1,internal)

                        if self.prebody:
                            self.prebody[1](step)

                            self.timer.singleShot(self.prebody[0],
                                                  measureandcontinue)
                        else:
                            measureandcontinue()
                except StopIteration:
                    if self.onfinished:
                        ret = self.onfinished(self)
                    else:
                        ret = True
                    self.ready = ret

                self._internal = internal
        self.timer.singleShot(0, internal)
    
    def pause(self):
        self.cont = False

    def resume(self):
        self.cont = True
        self.timer.singleShot(0,self._internal)
        
    def stop(self):
        self.cont = False

class Scan(ScanCore):
    ''' Do a scan over certain steps, calling prebody before measurement
    and body to do the actual measurement, and save it to filename name'''
    
    def __init__(self, body=None, steps=None, iterer=None, prebody=None,
                 onfinished=None, name='scan', estimate=0, save=True,
                 *args, **kwargs):
        # Initialize the PunchingBag as a QObject
        super(self.__class__, self).__init__(body=body, steps=steps, iterer=iterer,
                prebody=prebody, onfinished=onfinished, name=name, estimate=estimate,
                *args, **kwargs)
        self.frames = []
        if save:
            self.onfinished = saver_scan
        self.result.connect(self.collector)

    @Slot(object)
    def collector(self, measurement):
        self.frames.append(measurement)


class ScanSaveEvery(ScanCore):
    ''' Do a scan over certain steps, calling prebody before measurement
    and body to do the actual measurement, and save it to filename name'''
    
    def __init__(self, body=None, steps=None, iterer=None, prebody=None,
                 onfinished=None, name='scan', estimate=0, save=True,
                 *args, **kwargs):
        # Initialize the PunchingBag as a QObject
        super(self.__class__, self).__init__(body=body, steps=steps, iterer=iterer,
                prebody=prebody, onfinished=onfinished, name=name, estimate=estimate,
                *args, **kwargs)
        #self.frames = []
        if save:
            self.result_and_step.connect(self.saver)

    @Slot(object)
    def saver(self, res):
        saver_sep(fname=self.name, steps=[res[0]], frame=[res[1]])


class WaitScan(ScanCore):
    '''Create a scan that only create a pause'''
    def __init__(self, waittime=0):
        '''Waittime in seconds'''
        super().__init__()
        self.name = 'Wait {:.0f}s'.format(waittime)
        self.waittime = waittime
        self._estimate = waittime
    
    def start(self):
        self.start_time = time.time()
        def tmp():
            self.ready = True
        self.timer.singleShot(self.waittime*1e3, tmp)

  
class Instruct(QtGui.QMessageBox, ScanCore):
    '''Give an instruction to follow, can be used as a Scan'''
    def __init__(self, text=''):
        super().__init__()
        self.name = 'Instruct'
        self._estimate = 5
        self.setModal(False)
        self.setWindowTitle('Do something')    
        self.btn = self.addButton('I did that!', QtGui.QMessageBox.AcceptRole)
        self.btn.clicked.connect(self.closeEvent)
        self.setText(text)
        
    def start(self):
        self.start_time = time.time()
        self.show()

    @property
    def estimate(self):
        return self._estimate
        
    def closeEvent(self, event):
        self.ready = True


class ScanManager(QtCore.QObject):
    '''Run different Scans in a row
    
    Example:
    def testsave(steps):
        print('Saved to x, steps', steps)
        return True

    def testwait():
        print('Start sleep')
        time.sleep(0.1)
        print('End sleep')
    
    qu = ScanManager()
    qu.finished.connect(lambda signal: print('Finished:',signal))
    qu.append(WaitScan(waittime=1e3))
    qu.append(Instruct('Click me'))
    mes = Scan(body=lambda *args: testwait(), steps=np.arange(10), onfinished=testsave)
    qu.queue.put(mes)
    qu.run()
    '''
    finished = QtCore.Signal(bool)
    newitem = QtCore.Signal(object)
    nextitem = QtCore.Signal(object)

    def __init__(self, monitor_function=None):
        QtCore.QObject.__init__(self)
        #super(self.__class__, self).__init__(self)
        self.scans = []
        self.paused = False
        self.end_time = None
        self.start_time = 0
        self.monitor_function = monitor_function
        self.item = None


    def __repr__(self):
        return 'ScanManager containing {}/{} items, estimated time: {:.1f}/{:.1f}s'.format(
                        self.counter, len(self.scans), self.estimate(), self.estimate(total=True))

    def show(self):
        ret = [(nr, ['','Ready'][scan.ready], scan) for nr,scan 
                in enumerate(self.scans)]
        ret.append(['Total duration: {}/{}'])
        return ret

    def estimate(self, total=False):
        return sum([scan.estimate for scan in self.scans if total or not scan.ready])
           
    def append(self, item):
        self.scans.append(item)
        self.newitem.emit(item)
        
    def pop(self, n=1):
        items = [self.scans.pop() for i in range(n)]
        self.newitem.emit(items)
        return items

    def start(self, position=0):
        '''Run scans from number position'''
        self.counter = position
        self.start_time = time.time()
        self.cont = True
        self.paused = False
        self.do()

    def start_from_first_not_ready(self):
        self.start(self.first_not_ready())

    def first_not_ready(self):
        '''Get the index of the first scan that is not ready'''
        return np.min([i for i, scan in enumerate(self.scans) if not scan.ready])

    def do(self):
        if len(self.scans) > self.counter:
            if self.item:
                self.item.finished.disconnect()
            self.item = self.scans[self.counter]
            self.counter += 1
            self.item.finished.connect(self.onedone)
            if self.monitor_function:
                try:
                    self.item.result.connect(self.monitor_function)
                except AttributeError:
                    pass
            try:
                self.item.start()
            except RuntimeError:
                self.item.finished.emit('Already done, skipping:',
                        self.item.name)
        else:
            self.finished.emit(True)
            self.end_time = time.time()

    @Slot(bool)
    def onedone(self, done):
        if done or True:
            self.nextitem.emit(done)
            print('Scan done. File written to:', done)
            if self.cont:
                self.do()
        else:
            raise Exception('Task not done, but signal fired', done)

    def pause(self):
        self.cont = False
        [scan.pause() for scan in self.scans]

    def resume(self):
        self.cont = True
        self.item.resume()

    def clear(self):
        '''Remove all scans from queue'''
        self.scans.clear()

    def reset(self, only_notready=False):
        '''Reset all scans in line'''
        for scan in self.scans:
            if (not only_notready) or (not scan.ready):
                scan.reset()

    def reset_notready(self):
        '''Reset all scans which are not finished'''
        return self.reset(only_notready=True)

    def reset_from_frist_not_ready(self):
        '''Reset all scans below first not finished scan'''
        reset = False
        for scan in self.scans:
            if not scan.ready:
                reset = True
            if reset:
                scan.reset()
    
    def reset_from_number(self, n):
        for i in (range(n,len(self.scans))):
            self.scans[i].cont=False
            self.scans[i].reset()

    def cleanup(self):
        '''Disconnect all unused signals from onedone slot'''
        for scan in self.scans:
            try:
                scan.finished.disconnect()
            except TypeError:
                pass

    
class ScanManagerMonitor(QtGui.QTableWidget):
    """Create a window that shows all the current set scans"""
    timer = QtCore.QTimer()
    def __init__(self, QueueManager):
        super(self.__class__, self).__init__()
        self.finished_rows = 0
        self.qm = QueueManager
        self.setupUI()
        self.populate()
        self.show()
        self.timer.timeout.connect(self.populate)
        self.timer.start(200)

    def setupUI(self):
        self.setWindowTitle('ScanManager Monitor')
        #self.setDisabled(True)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(['Name', 'Duration (s)', 'Endtime'])

    #@Slot(object)
    def populate(self, obj=None):
        tasks = self.qm.scans
        rows = len(tasks)+1
        self.setRowCount(rows)
        for row, task in zip(range(rows), tasks):
            item = QtGui.QTableWidgetItem(task.name)
            self.setItem(row,0, item)
            duration = max(task.estimate,1e-10)
            self.setItem(row,1, QtGui.QTableWidgetItem('{:.1f}'.format(duration)))
            if task.start_time:
                if task.ready:
                    endtime = task.ready_time
                else:
                    endtime = time.time() + duration
                endtime = time.strftime('%H:%M:%S', time.localtime(endtime))
            else:
                endtime = ''
            self.setItem(row,2, QtGui.QTableWidgetItem(str(endtime)))
            if task.ready:
                self.colorRow(row, 'lightblue')
            else:
                self.colorRow(row, 'white')
        self.setItem(rows-1, 1, QtGui.QTableWidgetItem('End time'))
        #st = self.qm.start_time if self.qm.start_time else time.time()
        st = time.time()
        endtime = time.strftime('%H:%M:%S', time.localtime(self.qm.estimate() + st))
        self.setItem(rows-1, 2, QtGui.QTableWidgetItem(endtime))
        self.setVerticalHeaderLabels([str(i) for i in range(self.rowCount())])

        #self.resize(myGetQTableWidgetSize(self))
        
    def finish(self):
        self.colorRow(self.finished_rows, 'lightblue')
        self.finished_rows += 1
    
    def colorRow(self, rownumber, color):
        for col in range(self.columnCount()):
            try:
                self.item(rownumber, col).setBackgroundColor(QtGui.QColor(color))
            except AttributeError:
                pass

class ScanManagerMonitorWindow(QtGui.QWidget):
    def __init__(self, QueueManager, sound_file=''):
        super(self.__class__, self).__init__()
        self.smm = ScanManagerMonitor(QueueManager)

        self.setWindowTitle('ScanManager Monitor')

        self.layout = QtGui.QVBoxLayout()
       
        self.buttons()

        # Add the button box to the bottom of the main VBox layout
        self.layout.addLayout(self.button_box)
            
        self.layout.addWidget(self.smm)
              
        # Set the VBox layout as the window's main layout
        self.setLayout(self.layout)
        
        if sound_file:
            self.sound = QtGui.QSound(sound_file)
            self.smm.qm.finished.connect(self.sound.play)
        
    def buttons(self):
        self.button_box = QtGui.QHBoxLayout()

        self.btn_start = QtGui.QPushButton('Start from first unfinished')
        self.btn_pause = QtGui.QPushButton('Pause')
        self.btn_cleanup = QtGui.QPushButton('Reset unfinished')
        self.btn_reset = QtGui.QPushButton('Reset')
        self.btn_clear = QtGui.QPushButton('clear')
         # Add it to the button box
        self.button_box.addWidget(self.btn_start)
        self.button_box.addWidget(self.btn_pause)
        self.button_box.addWidget(self.btn_cleanup)
        self.button_box.addWidget(self.btn_reset)
        self.button_box.addWidget(self.btn_clear)
        # Add actions
        self.btn_start.clicked.connect(self.smm.qm.start_from_first_not_ready)
        self.btn_pause.clicked.connect(self.smm.qm.pause)
        self.btn_cleanup.clicked.connect(self.smm.qm.reset_notready)
        self.btn_reset.clicked.connect(self.smm.qm.reset)
        self.btn_clear.clicked.connect(self.smm.qm.clear)
        
### Windows for cameras
class Buttons(object):
    def __init__(self):
        # Create a horizontal box layout to hold the button
        self.button_box = QtGui.QHBoxLayout()

        # Add stretch to push the button to the far right
        self.button_box.addStretch(1)
        self.fname_box = QtGui.QLineEdit()
        self.btn_save = QtGui.QPushButton('Save')
        self.exposure_time =  QtGui.QDoubleSpinBox()
        self.exposure_time.setAccessibleName('exposure_time')
        self.gain =  QtGui.QDoubleSpinBox()
        self.gain.setMaximum(999)
        self.gain.setDecimals(0)
        self.gain.setAccessibleName('gain')
        self.btn_start = QtGui.QPushButton('Start')
        self.btn_stop = QtGui.QPushButton('Stop')
         # Add it to the button box
        self.button_box.addWidget(self.fname_box)
        self.button_box.addWidget(self.btn_save)
        self.button_box.addWidget(QtGui.QLabel('Exposure time'))
        self.button_box.addWidget(self.exposure_time)
        self.button_box.addWidget(self.gain)
        self.button_box.addWidget(self.btn_start)
        self.button_box.addWidget(self.btn_stop)
		
class Cam2_window(QtGui.QWidget, Buttons):
    ''' An example of PySide/PyQt absolute positioning; the main window
        inherits from QWidget, a convenient widget for an empty window. '''

    def __init__(self):
        # Initialize the object as a QWidget and
        # set its title and minimum width
        QtGui.QWidget.__init__(self)
        Buttons.__init__(self)
        self.setWindowTitle('Basler camera')
        self.setMinimumWidth(400)

        # Create the QVBoxLayout that lays out the whole form
        self.layout = QtGui.QVBoxLayout()
        
        # Add the button box to the bottom of the main VBox layout
        self.layout.addLayout(self.button_box)
            
        self.init_plot_area()
        
        self.layout.addWidget(self.win2)
              
        # Set the VBox layout as the window's main layout
        self.setLayout(self.layout)
        
        # Add button for ROI
        self.btn_autoROI = QtGui.QPushButton('Set ROI')
        self.button_box.addWidget(self.btn_autoROI)

        self.fname_box.returnPressed.connect(self.save)
        self.btn_save.clicked.connect(self.save)
        self.btn_autoROI.clicked.connect(self.roi_set_from_view)
    
    def save(self):
        text = self.fname_box.text()
        if len(text) > 0:
            text = '_' + text
        saver(frames2[-1], fname='{}_basler{}.fits'.format(find_available_number(), text))
        
    def init_plot_area(self):
        win2 = pg.GraphicsLayoutWidget()#title="Camera")
        win2.setWindowTitle('Camera')
        
        self.imv = pg.ImageItem(np.random.normal(size=(4096,3000)))
        imv = self.imv
        
        view = pg.ViewBox()
        view.setAspectLocked(True)
        view.addItem(imv)
        self.roi = pg.LineSegmentROI([[10,10], [100,100]], pen='r')
        self.roi.sigRegionChanged.connect(self.update_cut)
        view.addItem(self.roi)
        win2.addItem(view)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(imv)
        hist.setLevels(0, 2**8)  # 2**12)
        win2.addItem(hist)

        win2.nextRow()
        self.p2 = win2.addPlot(colspan=2)
        self.p2.setMaximumHeight(250)
        
        self.win2 = win2
        
    def run(self):
        # Show the form
        self.show()
    
    def update_plot(self, y):
        self.imv.setImage(y.data.T.astype(np.float), autoLevels=False)
        self.update_cut()
    
    def update_cut(self):
        if len(frames2)>0:
            selected = self.roi.getArrayRegion(frames2[-1].data.T, self.imv)
            self.p2.plot(selected, clear=True)
 
    def roi_set_from_view(self, *args):
        # These are still relative coordinates.
        view = self.imv.getViewBox()
        coords = view.viewRange()
        roi = cam2.calc_roi_from_rel_coords(coords[::-1])
        cam2.set_roi(*roi)
        view.setRange(xRange=(0,cam2.Width), yRange=(0,cam2.Height))

class Spectrometer_widget(pg.PlotItem):
    '''Display single line of spectrometer'''
    
    def __init__(self):
        pg.PlotItem.__init__(self)
        x = np.arange(512)
        y = np.random.normal(size=(512))
        self.curve = self.plot(x, y, pen='b')      
        self.avg = None
    
    def run(self):
        self.show()
    
    def update_curve(self, y):
        self.curve.setData(y)
        
    def update_data(self, y):
        if self.avg:
            try:
                length = len(frames)
                y2 = np.array([frames[i].data for i in range(length-self.avg, length)]).mean(axis=0)[0]
                self.update_curve(y2)
            except IndexError:
                self.update_curve(y.data[0])
        else:
            self.update_curve(y.data[0])


class Spectrometer_avg_widget(pg.GraphicsLayout):
    '''Create a widget that shows the last 110 measurements''' 
    def __init__(self):
        # Initialize the object as a QWidget and
        # set its title and minimum width
        pg.GraphicsLayout.__init__(self)
        
        self.setWindowTitle('Spectrometer Last 110')
            
        self.init_plot_area()

        self.buffer = collections.deque(maxlen=110)
        self.buffer.extend([np.zeros(512)]*110)
    
    def init_plot_area(self):
        self.imv = pg.ImageItem(np.random.normal(size=(512,110)), border='k')
        imv = self.imv
        imv.setBorder('k')
        
        view = pg.ViewBox()
        view.addItem(imv)
        self.view = view
        self.addItem(view)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(imv)
        hist.setLevels(0, 2**12)
        self.addItem(hist)
        
    def run(self):
        # Show the form
        self.show()
        # Run the qt application
    
    def update_data(self, y):
        # This takes less than 1ms on QO36
        self.buffer.append(y.data[0])
        data = np.array(self.buffer)
        self.imv.setImage(data.T.astype(np.float), autoLevels=False)


class Spectrometer_window(QtGui.QWidget):
    '''Window that shows the spectrum and save buttons''' 
    def __init__(self):
        # Initialize the object as a QWidget and
        # set its title and minimum width
        QtGui.QWidget.__init__(self)
               
        self.setWindowTitle('Spectrometer')
        self.setMinimumWidth(400)
              
        # Create the QVBoxLayout that lays out the whole form
        self.layout = QtGui.QVBoxLayout()
        # Set the VBox layout as the window's main layout
        self.setLayout(self.layout) 
                
        #Add buttons
        self.buttons = Buttons()
        self.layout.addLayout(self.buttons.button_box) #QHBoxLayout
        self.buttons.fname_box.returnPressed.connect(self.save)
        self.buttons.btn_save.clicked.connect(self.save)   
        
        self.layout_graphics = pg.GraphicsLayoutWidget()#border=(100,100,100))
        self.layout.addWidget(self.layout_graphics)

        self.widgets = [Spectrometer_widget()]
        self.layout_graphics.addItem(self.widgets[0])
      
    def save(self):
        text = self.buttons.fname_box.text()
        if len(text) > 0:
            text = '_' + text
        saver(frames[-1], fname='{}_spectr{}.fits'.format(find_available_number(), text))

    def run(self):
        # Show the form
        self.show()
        # Run the qt application

    def update_data(self, y):
        for widget in self.widgets:
            widget.update_data(y)
            

class Spectrometer_avg_window(Spectrometer_window):
    '''Window that shows the spectrum, a save button and the last 110 measurements''' 
    def __init__(self):
        Spectrometer_window.__init__(self)
        self.setWindowTitle('Spectrometer last 110')
        
        self.widgets.append(Spectrometer_avg_widget())
        self.layout_graphics.nextRow()
        self.layout_graphics.addItem(self.widgets[1])


class ShowDifference():
    '''Show the difference between the actual measurement, and some previous'''
    def __init__(s, imv, data):
        s.imv = imv
        # normalize
        s.data = data.T/data.max()

    def update_difference(s, y):
        recent = y.data.T.astype(np.float)
        # normalize
        recent = recent/recent.max()
        different = recent - s.data
        s.imv.setImage(different, autoLevels=True)
