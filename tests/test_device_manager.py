from labtools.device_manager import Devices, parse_feat_settings, lantz
import yaml

def test_device():
    config = {
        'devices': [
            {'name': 'newport_delay',
             'type': 'translation',
             'resource_id': {'type': 'serial',
                             'manufacturer': 'FTDI',
                             'pid': 24577,
                             'vid': 1027,
                             'serial_number': 'FT1LXFSK',

                             },
             'module': 'lantz.drivers.newport_motion',
             'driver': 'SMC100',
             },
        ]
        ,
    }
    #print('\n',config)
    devices = Devices(config)


def test_unit():
    res = parse_feat_settings({'backlash': 'unit 0.009 mm'})
    assert res == {'backlash': lantz.ureg.parse_expression('0.009 millimeter')}


def test_write_load(tmpdir):
    config = {
        'devices': [
            {'name': 'newport_delay',
             'type': 'translation',
             'resource_id': {'type': 'serial',
                             'manufacturer': 'FTDI',
                             'pid': 24577,
                             'vid': 1027,
                             'serial_number': 'FT1LXFSK',

                             },
             'module': 'lantz.drivers.newport_motion',
             'driver': 'SMC100',
             },
        ]
        ,
    }
    with open(tmpdir+'config.yml', 'w') as f:
        yaml.dump(config, f)
    #print('\n\n', type(config['devices'][0]['resource_id']['type']))

    with open(tmpdir+'config.yml', 'r') as f:
        config2 = yaml.load(f)
    #print(type(config2['devices'][0]['resource_id']['type']))

    assert config2['devices'][0]['resource_id']['type'] == config['devices'][0]['resource_id']['type']
    assert config['devices'][0]['resource_id']['type'] is 'serial'
    assert config2['devices'][0]['resource_id']['type'] == 'serial'
    devices = Devices(config2)