import logging
import importlib
import serial.tools.list_ports
import lantz
from copy import copy

def list_serial_ports():
    devices = serial.tools.list_ports.comports()
    attrs = ['description', 'device', 'device_path', 'hwid',
             'interface', 'location', 'manufacturer', 'name',
             'pid', 'product', 'serial_number', 'subsystem',
             'usb_device_path', 'usb_interface_path', 'vid']
    return {device.device:
                {attr: getattr(device, attr) for attr in attrs}
            for device in devices}


def find_device_VISA(fmt='ASRL{}::INSTR', **kwargs):
    """Find a USB to serial converter and format VISA identifier.
    Possible selection critria are added as kwarg and are:

    ['description', 'device', 'device_path', 'hwid',
     'interface', 'location', 'manufacturer', 'name',
     'pid', 'product', 'serial_number', 'subsystem',
     'usb_device_path', 'usb_interface_path', 'vid']

    Note: pid and vid are reported as hex numbers by dmesg

    Examples:
    find_device_VISA(manufacturer='Prolific Technology Inc.', pid=0x2303)
    find_device_VISA(device_path='/sys/devices/pci0000:00/0000:00:14.0/usb1/1-13/1-13.4/1-13.4.4/1-13.4.4.4/1-13.4.4.4:1.0/ttyUSB2',)
    """
    devices = serial.tools.list_ports.comports()
    matched = []
    for device in devices:
        try:
            for attr, value in kwargs.items():
                if getattr(device, attr) != value:
                    raise ValueError('Not matched')
        except ValueError:
            pass
        else:
            matched += [device]
    try:
        return fmt.format(matched[0].device)
    except IndexError:
        raise ValueError('No device found with ', kwargs)


def parse_feat_settings(feat_settings):
    """Makes pint units from every string starting with unit"""
    parsed = {}
    for feat in feat_settings:
        value = feat_settings[feat]
        try:
            if value.startswith('unit'):
                value = lantz.ureg.parse_expression(value.lstrip('unit '))
        except AttributeError:
            # Value is not a string
            pass
        print({feat: value})
        parsed.update({feat: value})
    return parsed


class Devices():
    '''Load a series of Lantz devices'''

    def __init__(self, config, logger=None):
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger('Device manager')
            self.logger.setLevel(logging.INFO)
        self.config = config
        self.devices = {}
        self.drivers = {}
        self.modules = {}
        self.load_imports()
        self.create_devices()

    def load_imports(self):
        self.logger.debug('Importing modules')
        for device in self.config['devices']:
            try:
                module = device['module']
                self.modules[module] = importlib.import_module(module)
                self.logger.debug('For device %s imported %s', device['name'], device['module'])
            except KeyError:
                pass  # Nothing to import

    def create_devices(self):
        self.logger.debug('Creating devices')
        for device in self.config['devices']:
            if device.get('alias', False):
                continue  # this is an alias, not load everything

            if isinstance(device['resource_id'], dict):
                resource_id = self.parse_resource_id(device['resource_id'])
            else:
                resource_id = device['resource_id']

            try:
                loaderfunc = getattr(self.modules[device['module']], device['driver'])
            except KeyError:
                loaderfunc = globals()[device['driver']]
            try:
                self.devices[device['name']] = loaderfunc(resource_id,
                                                          **device.get('driver_kwargs', {}))
                self.logger.debug('Loaded %s with resource id %s', device['name'], resource_id)
            except TypeError:
                raise TypeError('Error occured while loading {}', device['name'])

    def parse_resource_id(self, resource_id_dict):
        resource_id_dict_local = copy(resource_id_dict)
        type_ = resource_id_dict_local.pop('type')
        if type_ != 'serial':
            raise NotImplementedError('Parsing type {} is not implemented', type_)

        return find_device_VISA(**resource_id_dict_local)

    def initialize(self):
        lantz.initialize_many(self.devices.values())

        self.create_aliases()
        self.export_globals()

        for device in self.config['devices']:
            try:
                parsed = parse_feat_settings(device['feat_settings'])
                self.devices[device['name']].update(parsed)
            except KeyError:
                pass

    def finalize(self):
        for device in self.config['devices']:
            try:
                self.devices[device['name']].update(device['shutdown_pre']['feat_settings'])
            except (KeyError, AttributeError):
                # attribute error for not initialized devices
                pass
        lantz.finalize_many(self.devices.values())

    def create_aliases(self):
        for device in self.config['devices']:
            if not device.get('alias', False):
                continue

            self.devices[device['name']] = self.devices[device['variable']]
            if device.get('feat', None) is not None:
                self.devices[device['name']] = getattr(self.devices[device['name']], device['feat'])
            if device.get('index', None) is not None:
                self.devices[device['name']] = self.devices[device['name']][device['index']]

            self.logger.debug('Created alias %s with to', device['name'], device['variable'])

    def export_globals(self):
        globals().update(self.devices)
