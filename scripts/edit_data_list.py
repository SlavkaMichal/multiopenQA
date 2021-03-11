import jsonlines as jl

DUMP_LIST = 'data/wiki/dump_list.jsonl'


def update(key, values):
    lines = []
    with jl.open(DUMP_LIST) as reader:
        for line in reader:
            lines.append(line)
    if len(values) != len(lines):
        raise RuntimeError(f"Number of lines({len(lines)}) is not equal to number of updating values ({len(values)})!")

    with jl.open(DUMP_LIST, mode='w') as writer:
        for line, value in zip(lines, values):
            line[key] = value
            writer.write(line)


def remove(key):
    lines = []
    with jl.open(DUMP_LIST) as reader:
        for line in reader:
            lines.append(line)

    with jl.open(DUMP_LIST, mode='w') as writer:
        for line in lines:
            del line[key]
            writer.write(line)


def update_true(update, value, key, cmp_value):
    lines = []
    with jl.open(DUMP_LIST) as reader:
        for line in reader:
            lines.append(line)

    with jl.open(DUMP_LIST, mode='w') as writer:
        for line in lines:
            if type(cmp_value) == list:
                if line[key] in cmp_value:
                    line[update] = value
            else:
                if line[key] == cmp_value:
                    line[update] = value
            writer.write(line)


def return_true(key, cmp_value):
    lines = []
    with jl.open(DUMP_LIST) as reader:
        for line in reader:
            lines.append(line)
    ret_lines = []

    for line in lines:
        if type(cmp_value) == list:
            if line[key] in cmp_value:
                ret_lines.append(line)
        else:
            if line[key] == cmp_value:
                ret_lines.append(line)
    return ret_lines
