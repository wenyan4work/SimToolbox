# if the last frame of data is incomplete, trim the frame

import msgpack as mp


def get_step_offset(offset_file_name):
    offset_file = open(offset_file_name, 'r')
    step = 0
    offset = 0
    for line in offset_file:
        entry = line.split()
        if len(entry) == 2 and int(entry[0]) >= step and int(entry[1]) >= offset:
            step = int(entry[0])
            offset = int(entry[1])
        else:
            break
    return step, offset


def trim_data(data_file_name, step, offset):
    data_file = open(data_file_name, 'rb')
    data_file.seek(offset)
    unpacker = mp.Unpacker(data_file)
    frame = unpacker.unpack()
    msg = unpacker.unpack()
    if frame['stepID'] == step and msg == 'EOT':
        trunc_offset = offset+unpacker.tell()
        print(trunc_offset)
        data_file.close()
        data_file = open(data_file_name, 'r+')
        data_file.seek(trunc_offset)
        data_file.truncate()
        data_file.close()
    else:
        print('data does not match offset, data untouched')
        data_file.close()
    return


def unpack_verify(data_file_name):
    data_file = open(data_file_name, 'rb')
    unpacker = mp.Unpacker(data_file)
    for obj in unpacker:
        if type(obj) is dict:
            print(obj['stepID'])
        if type(obj) is str:
            print(obj)

    data_file.close()
    return


step, offset = get_step_offset('offset_r0.txt')
print(step, offset)
trim_data('data_r0.msgpack', step, offset)

unpack_verify('data_r0.msgpack')
